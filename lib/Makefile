TARGET := libsmvs.a
include ../Makefile.inc

CXXFLAGS += -I${MVE_ROOT}/libs -I../lib -msse4.2 -pthread
LDLIBS += -lpng -ltiff -ljpeg -pthread

SOURCES := $(wildcard [^_]*.cc)
${TARGET}: ${SOURCES:.cc=.o}
	$(AR) rcs $@ $^

_test%: _test%.o libmve_sfm.a libmve.a libmve_util.a
	${LINK.cc} -o $@ $^ ${LDLIBS}

clean:
	${RM} ${TARGET} *.o Makefile.dep

.PHONY: clean

