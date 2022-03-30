
SOURCES 	:= $(wildcard *.c)
OBJECTS 	:= $(SOURCES:.c=.o)
CFLAGS 		:= -c -fPIC -Wsign-conversion -Wsign-compare
CC 			:= clang
all: $(OBJECTS)

%.o: %.c *.h
	@ echo "Compiling:" $<
	@ echo "C compiler:" $(CC)
	@ echo "CFLAGS:" $(CFLAGS)
	@ $(CC) $(CFLAGS) $< -o $@

.PHONY: 
clean:
	@ rm -rf *.o
