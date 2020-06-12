#include <ulib.h>
#include <syscall.h>
#include <string.h>
#include <error.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#define printf(...)                     fprintf(1, __VA_ARGS__)

const char *strings[] = {
    "If you can read this, this is really bad",
    "Burn after reading this string, it is a secret string",
    "Congratulations, you just spied on an application",
    "Wow, you broke the security boundary between user space and kernel",
    "Welcome to the wonderful world of microarchitectural attacks",
    "Please wait while we steal your secrets...",
    "Don't panic... But your CPU is broken and your data is not safe",
    "How can you read this? You should not read this!"};

int main() {
    srand(sys_gettime());
    const char *secret = strings[rand() % (sizeof(strings) / sizeof(strings[0]))];
    int len = strlen(secret);
    printf("Secret: %s\n", secret);
    uint32_t vaddr = (uint32_t)secret;
    printf("Virtual address: 0x%x\n", vaddr);
    uint32_t paddr = sys_getpaddr(vaddr);
    printf("Kernel virtual address: 0x%x\n", paddr);
    int pid = fork();
    if (pid == 0) {
        // child process
        const char *argv[10];
        char addr[20];
        char size[20];
        snprintf(addr, sizeof(addr), "0x%x\n", paddr);
        snprintf(size, sizeof(size), "%d\n", 400);
        argv[0] = "spectre";
        argv[1] = addr;
        argv[2] = size;
        argv[3] = NULL;
        return __exec(NULL, argv);
    }
    else {
        while (1) {
            volatile size_t dummy = 0, i;
            for(i = 0; i < len; ++i) {
                dummy += secret[i];
            }
            yield();
        }
        int ret = 0;
        waitpid(pid, &ret);
    }
    return 0;
}