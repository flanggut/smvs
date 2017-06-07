#################################################
### Every *.cc file in this directory will be ###
### compiled into it's own app                ###
#################################################
TARGET = all
include ../Makefile.inc

CXXFLAGS += -I${MVE_ROOT}/libs -I../lib -msse4.2 -pthread -std=c++11
LDLIBS += -lpng -ltiff -ljpeg -lpthread

SOURCES := $(wildcard [^_]*.cc)
TOOLS := $(patsubst %.cc, tool_%, $(SOURCES))
BINARIES := $(patsubst tool_%, %, $(TOOLS))

all: $(SOURCES:.cc=.o) $(TOOLS)

.SECONDEXPANSION:
BIN = $(patsubst tool_%, %, $@)

tool_%: %.o libsmvs.a libmve_sfm.a libmve.a libmve_util.a
	${LINK.cc} -o $(BIN) $^ ${LDLIBS}

clean:
	${RM} ${BINARIES} *.o Makefile.dep

.PHONY: clean

