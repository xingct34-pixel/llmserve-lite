#include "serve.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#ifdef LLMSERVE_WITH_LLAMA
#include "llama.h"
#endif

namespace serve {
// Keep unfinished UTF-8 bytes until the next token completes the codepoint.
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
class MockEngine final:public Engine {
    Config c_;
public:
    explicit MockEngine(Config c):c_(std::move(c)){}
    std::string name() const override{return "mock";}
    Generation generate(const Request& r,const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
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
class LlamaEngine final:public Engine {
    Backend backend_;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model_{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx_{nullptr,llama_free};
    Config c_;

public:
    explicit LlamaEngine(Config c):c_(std::move(c)) {
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        model_.reset(llama_model_load_from_file(c_.model_path.c_str(),mp));
        if(!model_) throw std::runtime_error("cannot load GGUF model: "+c_.model_path);
        auto cp=llama_context_default_params();
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
        int n=llama_tokenize(vocab,formatted.data(),count,nullptr,0,true,true);
        if(n>=0) throw std::runtime_error("unexpected tokenizer size result");
        std::vector<llama_token> tokens(-n);
        n=llama_tokenize(vocab,formatted.data(),count,tokens.data(),int(tokens.size()),true,true);
        if(n<=0) throw std::runtime_error("tokenization failed");
        tokens.resize(n);
        if(n+r.max_tokens>int(llama_n_ctx(ctx_.get()))) throw std::invalid_argument("templated input plus output reservation exceeds context");
        auto abort=[](void* data)->bool{return (*static_cast<const std::function<bool()>*>(data))();};
        llama_set_abort_callback(ctx_.get(),abort,const_cast<std::function<bool()>*>(&stop));
        struct Reset {llama_context* ctx;~Reset(){llama_set_abort_callback(ctx,nullptr,nullptr);}} reset{ctx_.get()};
        for(int at=0;at<n;at+=128) {
            if(stop()) return {n,0,"cancelled"};
            auto batch=llama_batch_get_one(tokens.data()+at,std::min(128,n-at));
            if(llama_decode(ctx_.get(),batch)!=0) throw std::runtime_error("prefill decode failed or aborted");
        }
        std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(),llama_sampler_free);
        if(!sampler) throw std::runtime_error("sampler allocation failed");
        Generation g;g.prompt_tokens=n;
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
            if(i+1<r.max_tokens && llama_decode(ctx_.get(),llama_batch_get_one(&token,1))!=0) throw std::runtime_error("decode failed or aborted");
        }
        auto tail=complete_utf8(pending,true);
        if(!tail.empty()) emit(tail);
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
