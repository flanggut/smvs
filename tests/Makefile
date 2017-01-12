TARGET := test
include ../Makefile.inc
vpath gtest_main.a ${GTEST_PATH}/make/

SOURCES = $(wildcard gtest_*.cc)
INCLUDES = -I../lib -I${MVE_ROOT}/libs -I${GTEST_PATH}/include
CXXWARNINGS = -Wall -Wextra -pedantic -Wno-sign-compare
CXXFLAGS = -std=c++11 -pthread ${CXXWARNINGS} ${INCLUDES}
LDLIBS += -lpng -ltiff -ljpeg

test: ${SOURCES:.cc=.o} libsmvs.a libmve.a libmve_util.a gtest_main.a
	${LINK.cc} -o $@ $^ ${LDLIBS}

clean:
	${RM} ${TARGET} *.o Makefile.dep

.PHONY: test

