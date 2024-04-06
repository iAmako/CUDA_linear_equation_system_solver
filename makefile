CC=gcc
DEBUG=yes
ifeq ($(DEBUG),yes)
	CFLAGS=-W -Wall -pedantic -g
	LDFLAGS=
else
	CFLAGS= 
	LDFLAGS=
endif

EXEC=solver
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)

all: $(EXEC)

ifeq ($(DEBUG),yes)
	@echo "Generation en mode debug"
else
	@echo "Generation en mode release"
endif

solver: $(OBJ)
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