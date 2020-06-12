obj-m := syscall.o
KERNELDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
default: Spectre

all: Spectre module

Spectre: main.c
	gcc -O2 -msse2 $< -o Spectre
syscall_table.h: syscall_table.sh
	./syscall_table.sh
module: syscall_table.h syscall.c
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
clean: 
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions Module.symvers modules.order syscall_table.h Spectre