#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/unistd.h>
#include <asm/uaccess.h>
#include <linux/sched.h>
#include "syscall_table.h"

typedef unsigned long ul;
typedef unsigned long long ull;

// syscall defs
#define my_syscall_num 333
#define address_syscall_num 334

// mmap defs
#define DEV_MODULENAME "spectre"
#define DEV_CLASSNAME "spectreclass"
static int majorNumber;
static struct class *devClass = NULL;
static struct device *devDevice = NULL;

#ifndef VM_RESERVED
#define VM_RESERVED (VM_DONTEXPAND | VM_DONTDUMP)
#endif

#define OFFSET 4096
#define NUM_BYTES 256
#define MAX_SIZE ((OFFSET * NUM_BYTES))
unsigned int array1_size = 16;
char array1[160] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
char *array2 = NULL;
char temp = 0;

// syscall impls
unsigned int clear_and_return_cr0(void);
void setback_cr0(unsigned int val);
asmlinkage void sys_mycall(size_t);
asmlinkage size_t sys_myaddr(size_t);

int orig_cr0;
ul *sys_call_table = 0;
static int (*anything_saved_0)(void);
static int (*anything_saved_1)(void);

unsigned int clear_and_return_cr0(void)
{
	unsigned int cr0 = 0;
	unsigned int ret;
	asm("movq %%cr0, %%rax"
		: "=a"(cr0));
	ret = cr0;
	cr0 &= 0xfffffffffffeffff;
	asm("movq %%rax, %%cr0" ::"a"(cr0));
	return ret;
}

void setback_cr0(unsigned int val)
{
	asm volatile("movq %%rax, %%cr0" ::"a"(val));
}


// Shared kernel mem impl

struct mmap_info
{
	char *data;
	int reference;
};

static void
dev_vm_ops_open(struct vm_area_struct *vma)
{
	struct mmap_info *info;
	info = (struct mmap_info *)vma->vm_private_data;
	info->reference++;
}

static void
dev_vm_ops_close(struct vm_area_struct *vma)
{
	struct mmap_info *info;
	info = (struct mmap_info *)vma->vm_private_data;
	info->reference--;
}

static int
dev_vm_ops_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	struct page *page;
	struct mmap_info *info;

	info = (struct mmap_info *)vma->vm_private_data;
	if (info->data == NULL)
	{
		return 0;
	}

	page = virt_to_page((info->data) + (vmf->pgoff * PAGE_SIZE));

	get_page(page);
	vmf->page = page;

	return 0;
}

static const struct vm_operations_struct dev_vm_ops = {
	.open = dev_vm_ops_open,
	.close =
		dev_vm_ops_close,
	.fault = dev_vm_ops_fault,
};

int fops_mmap(struct file *filp, struct vm_area_struct *vma)
{
	unsigned long size = (unsigned long)(vma->vm_end - vma->vm_start);
	if (size > MAX_SIZE) {
		return -EINVAL;
	}
	vma->vm_ops = &dev_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	vma->vm_private_data = filp->private_data;
	dev_vm_ops_open(vma);
	return 0;
}

int fops_close(struct inode *inode, struct file *filp)
{
	if(filp->private_data != NULL) {
		struct mmap_info *info = filp->private_data;
		kfree(info);
		filp->private_data = NULL;
	}
	return 0;
}


int fops_open(struct inode *inode, struct file *p_file)
{
	struct mmap_info *info;

	info = kmalloc(sizeof(struct mmap_info), GFP_KERNEL);

	if (array2 == NULL)
	{
		return ENOMEM;
	}

	info->data = array2;

	p_file->private_data = info;
	return 0;
}

static const struct file_operations dev_fops = {
	.open = fops_open,
	.release = fops_close,
	.mmap =
		fops_mmap,
};

static int dev_uevent(struct device *dev, struct kobj_uevent_env *env) {
	add_uevent_var(env, "DEVMODE=%#o", 0777);
	return 0;
}

//// Module entry/exit

static int __init init_mod(void)
{
	int ret = 0;
	// setup syscall
	sys_call_table = (ul *)sys_call_table_adress;
	anything_saved_0 = (int (*)(void))(sys_call_table[my_syscall_num]);
	anything_saved_1 = (int (*)(void))(sys_call_table[address_syscall_num]);

	orig_cr0 = clear_and_return_cr0();
	sys_call_table[my_syscall_num] = (ul)&sys_mycall;
	setback_cr0(orig_cr0);
	orig_cr0 = clear_and_return_cr0();
	sys_call_table[address_syscall_num] = (ul)&sys_myaddr;
	setback_cr0(orig_cr0);

	// Try to dynamically allocate a major number for the device
	majorNumber = register_chrdev(0, DEV_MODULENAME, &dev_fops);
	if (majorNumber < 0)
	{
		printk(KERN_ALERT "Failed to register a major number.\n");
		return -EIO; // I/O error
	}

	// Register the device class
	devClass = class_create(THIS_MODULE, DEV_CLASSNAME);
	devClass->dev_uevent = dev_uevent;
	// Check for error and clean up if there is
	if (IS_ERR(devClass))
	{
		printk(KERN_ALERT "Failed to register device class.\n");
		ret = PTR_ERR(devClass);
		goto goto_unregister_chrdev;
	}

	// Create and register the device
	devDevice = device_create(devClass,
							  NULL, MKDEV(majorNumber, 0),
							  NULL,
							  DEV_MODULENAME);

	// Clean up if there is an error
	if (IS_ERR(devDevice))
	{
		printk(KERN_ALERT "Failed to create the device.\n");
		ret = PTR_ERR(devDevice);
		goto goto_class_destroy;
	}

	// Alloc shared memory
	array2 = kcalloc(NUM_BYTES * OFFSET, sizeof(char), GFP_KERNEL);
	if (array2 == NULL)
	{
		printk(KERN_ERR "insufficient memory\n");
		/* insufficient memory: you must handle this error! */
		goto goto_class_destroy;
	}

	printk(KERN_INFO "Module registered.\n");

	return ret;

// Error handling - using goto
goto_class_destroy:
	class_destroy(devClass);
goto_unregister_chrdev:
	unregister_chrdev(majorNumber, DEV_MODULENAME);

	return ret;
}

asmlinkage void sys_mycall(size_t x)
{
	volatile int z;
	asm volatile("clflush 0(%0)\n" : : "c"(&array1_size) : "eax");
	for (z = 0; z < 100; ++z) {}
	asm volatile("clflush 0(%0)\n" : : "c"(&x) : "eax");
	for (z = 0; z < 100; ++z) {}
	if (x < array1_size)
    {
        temp &= array2[array1[x] * OFFSET];
    }
}

asmlinkage size_t sys_myaddr(size_t x) {
	if(x == 1) {
		return (size_t)&array1;
	}
	return 0;
}

static void __exit exit_mod(void)
{
	orig_cr0 = clear_and_return_cr0();
	sys_call_table[my_syscall_num] = (ul)anything_saved_0;
	setback_cr0(orig_cr0);

	orig_cr0 = clear_and_return_cr0();
	sys_call_table[address_syscall_num] = (ul)anything_saved_1;
	setback_cr0(orig_cr0);

	device_destroy(devClass, MKDEV(majorNumber, 0));
	class_unregister(devClass);
	class_destroy(devClass);
	unregister_chrdev(majorNumber, DEV_MODULENAME);
	if (array2 != NULL)
	{
		kfree(array2);
	}
	printk(KERN_INFO "Module unregistered.\n");
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");
