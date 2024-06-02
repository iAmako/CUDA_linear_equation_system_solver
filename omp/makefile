CC=gcc
DEBUG=no

CFLAGS= -fopenmp -O2 -use_fast_math
LDFLAGS = -fopenmp -O2 -use_fast_math

ifeq ($(DEBUG),yes)
	CFLAGS+= -g 
endif

EXEC= solver_omp.exe

SRC_SOLV = $(filter-out main_system_solver.c, $(wildcard *.c))
OBJ_SOLV = $(SRC_SOLV:.c=.o)

all: $(EXEC)

ifeq ($(DEBUG),yes)
	@echo "Generation en mode debug"
else
	@echo "Generation en mode release"
endif

solver_omp.exe: $(OBJ_SOLV)
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