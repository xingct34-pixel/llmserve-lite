#include "serve.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#ifdef LLMSERVE_WITH_LLAMA
#include "llama.h"
#endif

namespace serve {
// 一个token可能只含中文字符的部分UTF-8字节；未凑齐就留在buffer，避免SSE发出半个字符。
// 首字节判定1~4字节长度，后续字节须以二进制10开头；非法或最终残缺序列替换为U+FFFD。
std::string complete_utf8(std::string& buffer,bool flush) {
    size_t pos=0;
    std::string out;
    while(pos<buffer.size()) {
        auto c=static_cast<unsigned char>(buffer[pos]);
        size_t width=c<0x80?1:((c&0xe0)==0xc0?2:((c&0xf0)==0xe0?3:((c&0xf8)==0xf0?4:0)));
        if(!width){out+="\xef\xbf\xbd";++pos;continue;}
        if(pos+width>buffer.size()) {if(flush){out+="\xef\xbf\xbd";pos=buffer.size();}break;}
        bool valid=true;
        for(size_t k=1;k<width;++k) if((static_cast<unsigned char>(buffer[pos+k])&0xc0)!=0x80) valid=false;
        if(valid && width>1) {
            const auto b=static_cast<unsigned char>(buffer[pos+1]);
            if((width==2 && c<0xc2)||(width==3 && ((c==0xe0&&b<0xa0)||(c==0xed&&b>=0xa0)))||
               (width==4 && (c>0xf4||(c==0xf0&&b<0x90)||(c==0xf4&&b>=0x90)))) valid=false;
        }
        if(!valid){out+="\xef\xbf\xbd";++pos;continue;}
        out.append(buffer,pos,width);pos+=width;
    }
    buffer.erase(0,pos);return out;
}
namespace {
size_t codepoints(const std::string& s) {
    return std::count_if(s.begin(),s.end(),[](unsigned char c){return (c&0xc0)!=0x80;});
}
// Mock只产生固定文本并模拟延迟，用于服务层测试；其片段数和耗时不能当成真实模型性能。
class MockEngine final:public Engine {
    Config c_;
public:
    explicit MockEngine(Config c):c_(std::move(c)){}
    std::string name() const override{return "mock";}
    Generation generate(const Request& r,const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        // 字符数+8只是Mock的上下文估算，真实引擎使用聊天模板后的实际分词结果。
        const int n=int(codepoints(r.prompt))+8;
        if(n+r.max_tokens>c_.context_tokens) throw std::invalid_argument("input plus output reservation exceeds context (mock codepoint estimate)");
        if(r.prompt=="__error__") throw std::runtime_error("injected mock failure");
        const std::vector<std::string> pieces={"你","好","，","这","是","测","试","。"};
        Generation g;g.prompt_tokens=n;
        const auto begin=Clock::now();
        for(int i=0;i<r.max_tokens;++i) {
            for(int waited=0;waited<c_.mock_delay_ms;waited+=5) {
                if(stop()) return g;
                std::this_thread::sleep_for(std::chrono::milliseconds(std::min(5,c_.mock_delay_ms-waited)));
            }
            if(stop()) return g;
            if(r.prompt=="__error_after__" && i==3) throw std::runtime_error("injected midstream mock failure");
            if(!emit(pieces[size_t(i)%pieces.size()])) return g;
            ++g.output_tokens;
        }
        g.decode_ms=millis(begin,Clock::now());return g;
    }
};
#ifdef LLMSERVE_WITH_LLAMA
struct Backend {Backend(){llama_backend_init();} ~Backend(){llama_backend_free();}};
// RAII管理资源：成员按声明的反序销毁，先context、再model、最后backend。
// 模型/context启动时创建一次；后续请求复用，但不会共享对话历史。
class LlamaEngine final:public Engine {
    Backend backend_;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model_{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx_{nullptr,llama_free};
    Config c_;

public:
    explicit LlamaEngine(Config c):c_(std::move(c)) {
        // CPU首版不把任何模型层放到GPU计算；GGUF中的量化权重由llama.cpp加载执行。
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        model_.reset(llama_model_load_from_file(c_.model_path.c_str(),mp));
        if(!model_) throw std::runtime_error("cannot load GGUF model: "+c_.model_path);
        auto cp=llama_context_default_params();
        // n_ctx是上下文容量；128/64限制逻辑批/内部微批的token量，不代表128个并发请求。
        cp.n_ctx=c_.context_tokens;cp.n_batch=128;cp.n_ubatch=64;
        cp.n_threads=c_.threads;cp.n_threads_batch=c_.threads;
        ctx_.reset(llama_init_from_model(model_.get(),cp));
        if(!ctx_) throw std::runtime_error("cannot allocate llama context");
        if(!llama_model_chat_template(model_.get(),nullptr)) throw std::runtime_error("model has no chat template");
    }
    std::string name() const override{return "llama";}
    Generation generate(const Request& r,const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        const auto begin=Clock::now();
        // 清空上一请求的KV等上下文状态，保持请求隔离；这不等于重新加载模型权重。
        llama_memory_clear(llama_get_memory(ctx_.get()),true);
        llama_chat_message msg={"user",r.prompt.c_str()};
        const char* tmpl=llama_model_chat_template(model_.get(),nullptr);
        std::vector<char> formatted(r.prompt.size()*2+1024);
        int count=llama_chat_apply_template(tmpl,&msg,1,true,formatted.data(),int(formatted.size()));
        if(count<0) throw std::runtime_error("unsupported chat template");
        if(count>int(formatted.size())) {
            formatted.resize(count+1);
            count=llama_chat_apply_template(tmpl,&msg,1,true,formatted.data(),int(formatted.size()));
            if(count<0 || count>int(formatted.size())) throw std::runtime_error("chat template formatting failed");
        }
        auto vocab=llama_model_get_vocab(model_.get());
        // 先传空缓冲查询所需token数（不足时返回负数），按-n分配后再真正分词。
        int n=llama_tokenize(vocab,formatted.data(),count,nullptr,0,true,true);
        if(n>=0) throw std::runtime_error("unexpected tokenizer size result");
        std::vector<llama_token> tokens(-n);
        n=llama_tokenize(vocab,formatted.data(),count,tokens.data(),int(tokens.size()),true,true);
        if(n<=0) throw std::runtime_error("tokenization failed");
        tokens.resize(n);
        // 容量校验：模板化输入token数+最大输出预留 <= 实际context容量。
        if(n+r.max_tokens>int(llama_n_ctx(ctx_.get()))) throw std::invalid_argument("templated input plus output reservation exceeds context");
        auto abort=[](void* data)->bool{return (*static_cast<const std::function<bool()>*>(data))();};
        llama_set_abort_callback(ctx_.get(),abort,const_cast<std::function<bool()>*>(&stop));
        // stop回调的地址只在本次调用中有效；RAII在正常返回和抛异常时都解除绑定。
        struct Reset {llama_context* ctx;~Reset(){llama_set_abort_callback(ctx,nullptr,nullptr);}} reset{ctx_.get()};
        // Prefill：分块处理整段输入并建立KV缓存；每块之间可响应取消。
        for(int at=0;at<n;at+=128) {
            if(stop()) return {n,0,"cancelled"};
            auto batch=llama_batch_get_one(tokens.data()+at,std::min(128,n-at));
            if(llama_decode(ctx_.get(),batch)!=0) throw std::runtime_error("prefill decode failed or aborted");
        }
        // 贪心采样每次选当前最高分token；EOG表示模型主动结束，不计入输出token数。
        std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(),llama_sampler_free);
        if(!sampler) throw std::runtime_error("sampler allocation failed");
        Generation g;g.prompt_tokens=n;
        // 此处prefill_ms还含清缓存、模板、分词与sampler准备，不是单独矩阵计算时间。
        const auto decode_begin=Clock::now();g.prefill_ms=millis(begin,decode_begin);
        std::string pending;
        for(int i=0;i<r.max_tokens;++i) {
            if(stop()) return g;
            auto token=llama_sampler_sample(sampler.get(),ctx_.get(),-1);
            if(llama_vocab_is_eog(vocab,token)) {g.finish_reason="stop";break;}
            std::vector<char> piece(64);
            int bytes=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,false);
            if(bytes<0){piece.resize(-bytes);bytes=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,false);}
            if(bytes<0) throw std::runtime_error("token to piece failed");
            pending.append(piece.data(),bytes);++g.output_tokens;
            auto text=complete_utf8(pending);
            if(!text.empty() && !emit(text)) return g;
            // 将本次生成的token送回模型计算下一步；已达输出上限时不再做无用的下一次计算。
            if(i+1<r.max_tokens && llama_decode(ctx_.get(),llama_batch_get_one(&token,1))!=0) throw std::runtime_error("decode failed or aborted");
        }
        auto tail=complete_utf8(pending,true);
        if(!tail.empty()) emit(tail);
        // decode_ms含采样、文本转换和emit背压等待，不能解释为纯CPU算子执行时间。
        g.decode_ms=millis(decode_begin,Clock::now());return g;
    }
};
#endif
}
std::unique_ptr<Engine> make_engine(const Config& c) {
    if(c.engine=="mock") return std::make_unique<MockEngine>(c);
#ifdef LLMSERVE_WITH_LLAMA
    return std::make_unique<LlamaEngine>(c);
#else
    throw std::runtime_error("llama backend not compiled; configure with --preset cpu");
#endif
}
}
