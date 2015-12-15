CC = gcc
CFLAGS = -std=c11 -O2 -fopenmp -lm -g
ann: ann.c
	$(CC) $(CFLAGS) -o ann utilities.c utilities.h ann.c

test_utilities: test_utilities.c
	$(CC) $(CFLAGS) -o test_utilities utilities.c utilities.h test_utilities.c

clean:
	rm -f ann test_utilities