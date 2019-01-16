#!/bin/sh

NOW=$(date +"%F-%H-%M-%S")
FILENAME=C++STL_$1_$2_$NOW.txt

echo -e "Did $3 runs of Benchmark $1, Size $2 on $NOW using the C++ STL version-------------------------------------------------------"

for ((i=0; i < $3; i++));
do
	./$1.$2
done
