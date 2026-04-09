CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic

all: mlsys

mlsys: mlsys.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f mlsys
