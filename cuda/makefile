CC=nvcc
DEBUG=no

CFLAGS=
LDFLAGS=

ifeq ($(DEBUG),yes)
	CFLAGS+= -g 
endif

EXEC= solver_cuda.exe

SRC_SOLV = $(filter-out main_system_solver.cu, $(wildcard *.cu))
OBJ_SOLV = $(SRC_SOLV:.cu=.o)

all: $(EXEC)

ifeq ($(DEBUG),yes)
	@echo "Generation en mode debug"
else
	@echo "Generation en mode release"
endif

solver_cuda.exe: $(OBJ_SOLV)
	@$(CC) -o $@ $^ $(LDFLAGS)
	
%.o: %.cu
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