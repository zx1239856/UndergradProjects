#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <time.h>
#include <stdarg.h>

#define DEFAULT_PHYSICAL_OFFSET 0xffff880000000000ull

size_t inline phys_to_virt(size_t addr, size_t offset) {
    if(addr + offset < offset) {
        return addr;
    }
    return addr + offset;
}

// call this function to start a nanosecond-resolution timer
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

// call this function to end a timer, returning nanoseconds elapsed as a long
long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

typedef enum
{
    ERROR,
    INFO,
    SUCCESS
} d_sym_t;

static void print(d_sym_t symbol, const char *fmt, ...)
{
    switch (symbol)
    {
    case ERROR:
        printf("\x1b[31;1m[ERROR]\x1b[0m ");
        break;
    case INFO:
        printf("\x1b[33;1m[INFO]\x1b[0m ");
        break;
    case SUCCESS:
        printf("\x1b[32;1m[SUCCESS]\x1b[0m ");
        break;
    default:
        break;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
}

#endif