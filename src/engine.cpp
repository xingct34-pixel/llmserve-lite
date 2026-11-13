#include "serve.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#ifdef LLMSERVE_WITH_LLAMA
#include "llama.h"
#endif

namespace serve {


// 模块：UTF-8 文本边界处理（complete_utf8）
// 作用：缓存跨 token 的 UTF-8 字节，只吐出完整字符，并把非法序列替换为 U+FFFD（�）。
// 通俗讲：一个汉字占 3 个字节，而模型的一个 token 可能只输出了其中 1~2 个字节。
//         如果直接发给客户端，会看到乱码或半个字。所以没凑齐的字节先留在 buffer 里，下次再拼。
// 规则：首字节决定该字符占 1~4 字节；后续字节必须以二进制 10 开头；
//       非法序列，或结束时仍残缺的序列，用 U+FFFD 替换。
// 参数 flush：为 true 表示生成已结束，不会再有后续字节，残缺部分也要处理掉。


std::string complete_utf8(std::string& buffer, bool flush) {
    size_t pos = 0;
    std::string out;
    while (pos < buffer.size()) {
        auto c = static_cast<unsigned char>(buffer[pos]);
        // 根据首字节的高位判断字符宽度：
        // 0xxxxxxx 为 1 字节（ASCII）；110xxxxx 为 2 字节；1110xxxx 为 3 字节；11110xxx 为 4 字节；
        // 其余（如孤立的 10xxxxxx）不是合法首字节，宽度记 0
        size_t width = c < 0x80 ? 1
                                : ((c & 0xe0) == 0xc0
                                       ? 2
                                       : ((c & 0xf0) == 0xe0 ? 3 : ((c & 0xf8) == 0xf0 ? 4 : 0)));
        // 非法首字节：输出替换字符，跳过这 1 个字节继续
        if (!width) {
            out += "\xef\xbf\xbd";
            ++pos;
            continue;
        }
        // 字节还没收齐：生成未结束就先留着等下次；已结束则只能当作残缺，替换后丢弃
        if (pos + width > buffer.size()) {
            if (flush) {
                out += "\xef\xbf\xbd";
                pos = buffer.size();
            }
            break;
        }
        // 检查后续字节是否都以 10 开头
        bool valid = true;
        for (size_t k = 1; k < width; ++k)
            if ((static_cast<unsigned char>(buffer[pos + k]) & 0xc0) != 0x80)
                valid = false;
        // 额外排除"过长编码""代理区""超出 Unicode 上限"等非法情形
        if (valid && width > 1) {
            const auto b = static_cast<unsigned char>(buffer[pos + 1]);
            if ((width == 2 && c < 0xc2) ||                                    // 2 字节过长编码
                (width == 3 && ((c == 0xe0 && b < 0xa0) || (c == 0xed && b >= 0xa0))) ||  // 3 字节过长/代理区
                (width == 4 && (c > 0xf4 || (c == 0xf0 && b < 0x90) || (c == 0xf4 && b >= 0x90))))  // 4 字节过长/超上限
                valid = false;
        }
        if (!valid) {
            out += "\xef\xbf\xbd";
            ++pos;
            continue;
        }
        // 合法：整个字符原样输出
        out.append(buffer, pos, width);
        pos += width;
    }
    // 已处理的部分从 buffer 删除，剩下的（残缺字节）留待下次
    buffer.erase(0, pos);
    return out;
}

namespace {


// 模块：模拟计数（codepoints）
// 作用：仅供 Mock 引擎估算输入有多少个字符。
// 原理：UTF-8 的后续字节都以 10 开头，不以 10 开头的字节才是一个字符的开头，数它们就是字符数。


size_t codepoints(const std::string& s) {
    return std::count_if(s.begin(), s.end(), [](unsigned char c) {
        return (c & 0xc0) != 0x80;
    });
}


// 模块：模拟引擎（MockEngine）
//
