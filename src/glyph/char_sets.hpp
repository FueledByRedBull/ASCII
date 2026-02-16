#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ascii {

namespace CharSet {

const std::string BASIC = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const std::string BLOCKS = " \xE2\x96\x91\xE2\x96\x92\xE2\x96\x93\xE2\x96\x88\xE2\x96\x80\xE2\x96\x84";
const std::string LINE_ART = "-|/\\_`\xC2\xB4\xE2\x94\x80\xE2\x94\x82\xE2\x94\x8C\xE2\x94\x90\xE2\x94\x94\xE2\x94\x98\xE2\x94\x9C\xE2\x94\xA4\xE2\x94\xAC\xE2\x94\xB4\xE2\x94\xBC";

inline bool is_valid_continuation_byte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

inline std::vector<uint32_t> to_codepoints(const std::string& s) {
    std::vector<uint32_t> result;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        unsigned char c = static_cast<unsigned char>(s[i]);
        
        if (c < 0x80) {
            cp = c;
            ++i;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= s.size() || !is_valid_continuation_byte(s[i+1])) {
                ++i;
                continue;
            }
            cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= s.size() || 
                !is_valid_continuation_byte(s[i+1]) || 
                !is_valid_continuation_byte(s[i+2])) {
                ++i;
                continue;
            }
            cp = ((c & 0x0F) << 12) | 
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 6) | 
                 (static_cast<unsigned char>(s[i+2]) & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= s.size() || 
                !is_valid_continuation_byte(s[i+1]) || 
                !is_valid_continuation_byte(s[i+2]) ||
                !is_valid_continuation_byte(s[i+3])) {
                ++i;
                continue;
            }
            cp = ((c & 0x07) << 18) | 
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 12) | 
                 ((static_cast<unsigned char>(s[i+2]) & 0x3F) << 6) | 
                 (static_cast<unsigned char>(s[i+3]) & 0x3F);
            i += 4;
        } else {
            ++i;
            continue;
        }
        
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            continue;
        }
        
        result.push_back(cp);
    }
    return result;
}

inline std::vector<uint32_t> get_set(const std::string& name) {
    if (name == "basic") return to_codepoints(BASIC);
    if (name == "blocks") return to_codepoints(BLOCKS);
    if (name == "line-art") return to_codepoints(LINE_ART);
    return to_codepoints(BASIC);
}

}

}
