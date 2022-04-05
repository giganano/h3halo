
SOURCES 	:= $(wildcard *.c)
OBJECTS 	:= $(SOURCES:.c=.o)
CFLAGS 		:= -c -fPIC -Wsign-conversion -Wsign-compare
CC 			:= clang
all: likelihood.o

%.o: %.c *.h
	@ echo "Compiling:" $<
	@ echo "C compiler:" $(CC)
	@ echo "CFLAGS:" $(CFLAGS)
	@ $(CC) $(CFLAGS) $< -o $@

.PHONY: 
clean:
	@ rm -rf __pycache__
	@ rm -f _fit_driver.c
	@ rm -rf build
	@ rm -rf *.so
	@ rm -rf *.o
