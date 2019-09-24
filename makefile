CC=gcc
CFLAGS=-I.

all: dilation

dilation: dilation.o
	$(CC) -o dilation dilation.o -I.
	rm dilation.o