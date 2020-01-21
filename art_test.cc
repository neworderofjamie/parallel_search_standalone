// C++ standard includes
#include <map>

// C standard includes
#include <cmath>
#include <cstdint>
#include <iostream>

// Include nightmare array
#include "big_array.h"



struct ARTNode
{
    std::map<uint8_t, ARTNode> children;
    bool leaf = false;
};


void add(uint32_t value, ARTNode &node)
{
    union {
        uint8_t bytes[4];
        uint32_t word;
    } valueUnion;

    valueUnion.word = value;

    node.children[valueUnion.bytes[0]].children[valueUnion.bytes[1]].children[valueUnion.bytes[2]].children[valueUnion.bytes[3]].leaf = true;
}

void print(const ARTNode &node)
{
    for(const auto &c : node.children) {
        std::cout << (int)c.first << ":";
        print(c.second);
        std::cout <<  std::endl;
    }
}
int main()
{
    const unsigned int numGroups = sizeof(mergedPresynapticUpdateGroupStartID0) / sizeof(unsigned int);

    ARTNode root;

    for(unsigned int i = 0; i < numGroups; i++) {
        const uint32_t g = mergedPresynapticUpdateGroupStartID0[i];
        add(g, root);
    }


    print(root);
    return EXIT_SUCCESS;
}
