TARGET := smvsrecon
include ../Makefile.inc

CXXFLAGS += -I${MVE_ROOT}/libs -I../lib -msse4.1 -pthread -std=c++11
LDLIBS += -lpng -ltiff -ljpeg -lpthread

SOURCES := $(wildcard [^_]*.cc)
${TARGET}: ${SOURCES:.cc=.o} libsmvs.a libmve.a libmve_util.a
	$(LINK.o) $^ $(LDLIBS) -o $@

_test%: _test%.o libsmvs.a libmve.a libmve_util.a
	${LINK.cc} -o $@ $^ ${LDLIBS}

clean:
	${RM} ${TARGET} *.o Makefile.dep

.PHONY: clean

