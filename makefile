CC=gcc
DEBUG=yes
ifeq ($(DEBUG),yes)
	CFLAGS=-W -Wall -pedantic -g
	LDFLAGS=
else
	CFLAGS= 
	LDFLAGS=
endif

EXEC=solver generator
SRC_SOLV = $(filter-out main_system_generator.c, $(wildcard *.c))
OBJ_SOLV = $(SRC_SOLV:.c=.o)

SRC_GEN = $(filter-out main_system_solver.c, $(wildcard *.c))
OBJ_GEN = $(SRC_GEN:.c=.o)

all: $(EXEC)

ifeq ($(DEBUG),yes)
	@echo "Generation en mode debug"
else
	@echo "Generation en mode release"
endif

solver: $(OBJ_SOLV)
	@$(CC) -o $@ $^ $(LDFLAGS)

generator: $(OBJ_GEN) 
	@$(CC) -o $@ $^ $(LDFLAGS)
	
%.o: %.c
	@$(CC) -o $@ -c $< $(CFLAGS)

.PHONY: clean mrproper

clean:
	@echo definition de clean
	rm -rf /F *.o
	@echo -------------------------------
mrproper: clean
	@echo cleaning
	rm -rf /F $(EXEC)
	@echo -------------------------------