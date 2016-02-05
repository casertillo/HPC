# Makefile

EXE1=lbm.exe
EXES=$(EXE1)

CC=mpicc
CFLAGS=-Wall #-DDEBUG
CFLAGS+=-O3
CLFLAGS=-lm

all: $(EXES)

$(EXES): lbm.o simulation.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLFLAGS)

.PHONY: all clean

clean:
	\rm -f $(EXES)

