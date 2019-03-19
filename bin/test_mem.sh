#!/bin/sh

NOW=$(date +"%F-%H-%M-%S")
FILENAME=C++STL_$1_$2_$NOW.txt

echo -e "Did $3 runs of Benchmark $1, Size $2 on $NOW using the C++ STL version-------------------------------------------------------"

export TBB_NUM_THREADS=56
for ((i=0; i < $3; i++));
do
	/usr/bin/time --verbose ./$1.$2
done
