#ifndef __COLOR_PRINT_H
#define __COLOR_PRINT_H

#include <stdio.h>
#include <stdarg.h>

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