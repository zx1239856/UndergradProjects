#include <stdlib.h>
#include <x86.h>

#define ULONG_MAX 4294967295u

int isspace(int c) {
    return c == ' ';
}

int isupper(int c) {
    return c >= 'A' && c <= 'Z';
}

int isdigit(int c) {
    return c >= '0' && c <= '9';
}

int isalpha(int c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

unsigned long strtoul(const char *str, char **endptr, int base)
{
	const char *s = str;
	unsigned long acc;
	int c;
	unsigned long cutoff;
	int neg = 0, any, cutlim;

	/*
	 * See strtol for comments as to the logic used.
	 */
	do {
		c = *s++;
	} while (isspace(c));
	if (c == '-') {
		neg = 1;
		c = *s++;
	} else if (c == '+')
		c = *s++;
	if ((base == 0 || base == 16) &&
	    c == '0' && (*s == 'x' || *s == 'X')) {
		c = s[1];
		s += 2;
		base = 16;
	}
	if (base == 0)
		base = c == '0' ? 8 : 10;
	cutoff = (unsigned long)ULONG_MAX / (unsigned long)base;
	cutlim = (unsigned long)ULONG_MAX % (unsigned long)base;
	for (acc = 0, any = 0;; c = *s++) {
		if (isdigit(c))
			c -= '0';
		else if (isalpha(c))
			c -= isupper(c) ? 'A' - 10 : 'a' - 10;
		else
			break;
		if (c >= base)
			break;
		if (((any < 0) || (acc > cutoff)) || ((acc == cutoff) && (c > cutlim)))
			any = -1;
		else {
			any = 1;
			acc *= base;
			acc += c;
		}
	}
	if (any < 0) {
		acc = ULONG_MAX;
	} else if (neg)
		acc = -acc;
	if (endptr != 0)
		*endptr = (any) ? (char *)(s - 1) : (char *)str;
	return (acc);
}