all: cudnnwrap.so

cudnnwrap.so: cudnnwrap.cpp
	g++ $< -o $@ -shared -O2 -fPIC

clean:
	rm -f cudnnwrap.so

.PHONY: all clean
