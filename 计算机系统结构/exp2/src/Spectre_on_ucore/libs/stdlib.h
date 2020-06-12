#ifndef __LIBS_STDLIB_H__
#define __LIBS_STDLIB_H__

#include <defs.h>

/* the largest number rand will return */
#define RAND_MAX    2147483647UL

/* libs/rand.c */
int rand(void);
void srand(unsigned int seed);
unsigned long strtoul(const char *str, char **endptr, int base);

/* libs/hash.c */
uint32_t hash32(uint32_t val, unsigned int bits);

#endif /* !__LIBS_RAND_H__ */

