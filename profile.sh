#!/bin/bash

# if any argument is provided, we will only run the png
# generation portion
if [ $# -eq 0 ]
  then
	python3 -m cProfile -o test.pstats main.py
fi
gprof2dot -f pstats test.pstats > test.dot
dot test.dot -Tpng -o call.png
rm test.dot test.pstats


