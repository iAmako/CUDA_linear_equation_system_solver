CC=gcc
DEBUG=yes

CFLAGS=-W -Wall -pedantic 


ifeq ($(DEBUG),yes)
	CFLAGS+= -g 
else
	CFLAGS += -O2 -fopenmp
	LDFLAGS = -fopenmp
endif

EXEC=solver.exe generator.exe
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

solver.exe: $(OBJ_SOLV)
	@$(CC) -o $@ $^ $(LDFLAGS)

generator.exe: $(OBJ_GEN) 
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