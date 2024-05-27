CC=nvcc
DEBUG=no

CFLAGS= #-Xptxas -O2 -use_fast_math
LDFLAGS = #-Xptxas -O2 -use_fast_math

ifeq ($(DEBUG),yes)
	CFLAGS+= -g 
endif

EXEC= solver_cuda.exe #solver.exe# generator.exe 
#SRC_SOLV = $(filter-out main_system_generator.c, $(wildcard *.c))
#OBJ_SOLV = $(SRC_SOLV:.c=.o)

#SRC_GEN = $(filter-out main_system_solver.c, $(wildcard *.c))
#OBJ_GEN = $(SRC_GEN:.c=.o)

SRC_SOLV = $(filter-out main_system_solver_cuda.cu, $(wildcard *.c*))
OBJ_SOLV = $(SRC_SOLV:.c=.o)


all: $(EXEC)

ifeq ($(DEBUG),yes)
	@echo "Generation en mode debug"
else
	@echo "Generation en mode release"
endif

#solver.exe: $(OBJ_SOLV)
#	@$(CC) -o $@ $^ $(LDFLAGS)

solver_cuda.exe: $(OBJ_SOLV)
	@$(CC) -o $@ $^ $(LDFLAGS)

#generator.exe: $(OBJ_GEN) 
#	@$(CC) -o $@ $^ $(LDFLAGS)
	
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