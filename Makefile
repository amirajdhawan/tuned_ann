CC = gcc
CFLAGS = -std=c11 -fopenmp -g
ann: ann.c
	$(CC) $(CFLAGS) -o ann utilities.c utilities.h ann.c -lm

test_utilities: test_utilities.c
	$(CC) $(CFLAGS) -o test_utilities utilities.c utilities.h test_utilities.c -lm

clean:
	rm -f ann test_utilities
