#!/bin/sh

addr=$(sudo cat /boot/System.map-$(uname -r) | grep " sys_call_table$" | cut -d" " -f1)

cat > syscall_table.h <<EOF
#ifndef SYS_CALL_TABLE_H
#define SYC_CALL_TABLE_H

#define sys_call_table_adress 0x$addr

#endif
EOF
