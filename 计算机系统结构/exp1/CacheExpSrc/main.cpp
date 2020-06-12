#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "CacheSim.h"
#include "argparse.hpp"

using namespace std;

int main(const int argc, const char *argv[])
{

    //以下为固定的测试
    ArgumentParser parser;

    parser.addArgument("-i", "--input", 1, false); //trace的地址
    parser.addArgument("--l1", 1, true);
    parser.addArgument("--l2", 1, true);
    parser.addArgument("--line_size", 1, true);
    parser.addArgument("--ways", 1, true);
    parser.addArgument("--mode", 1, true);
    parser.parse(argc, argv);

    CacheSim cache;
    /**cache有关的参数*/
    _u64 line_size[] = {64}; //cache line 大小：64B
    //_u64 ways[] = {16,32}; //组相联的路数
    _u64 ways[] = {32}; //组相联的路数
    //_u64 cache_size[] = {0x400000,0x800000};//cache大小:4M, 8M
    _u64 cache_size[] = {0x400000}; //cache大小:4M
    //cache_swap_style swap_style[] = {CACHE_SWAP_RAND, CACHE_SWAP_LRU, CACHE_SWAP_FRU, CACHE_SWAP_SRRIP, CACHE_SWAP_SRRIP_FP, CACHE_SWAP_BRRIP, CACHE_SWAP_DRRIP};
    cache_swap_style swap_style[] = {CACHE_SWAP_VICTIM, CACHE_SWAP_VIC_SHIP};
    int ms[] = {3};

    int i, j, m, k, n;
    for (m = 0; m < sizeof(cache_size) / sizeof(_u64); m++)
    {
        for (i = 0; i < sizeof(line_size) / sizeof(_u64); i++)
        {
            for (j = 0; j < sizeof(ways) / sizeof(_u64); j++)
            {
                for (k = 0; k < sizeof(ms) / sizeof(int); k++)
                {
                    for (n = 0; n < sizeof(swap_style) / sizeof(cache_swap_style); n++)
                    {
                        _u64 temp_cache_size, temp_line_size, temp_ways;
                        cache_swap_style temp_swap_style;

                        temp_cache_size = cache_size[m];
                        temp_line_size = line_size[i];
                        temp_ways = ways[j];
                        temp_swap_style = swap_style[n];
                        cache.init(temp_cache_size, temp_line_size, temp_ways, temp_swap_style);
                        cache.set_M(ms[k]);
                        string src = parser.retrieve<string>("mode");
                        if (src == "CPU")
                        {
                            cache.set_source(CPU_ONLY);
                        }
                        else if (src == "GPU")
                        {
                            cache.set_source(GPU_ONLY);
                        }
                        cache.load_trace(parser.retrieve<string>("input").c_str());
                        cache.re_init();
                    }
                }
            }
        }
    }
    return 0;
}
