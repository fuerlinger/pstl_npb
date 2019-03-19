#!/bin/sh

NOW=$(date +"%F-%H-%M-%S")
FILENAME=C++STL_$1_$2_$NOW.txt

echo -e "Did $3 runs of Benchmark $1, Size $2 on $NOW using the C++ STL version. Thread count is [1,64]-------------------------------------------------------"

export TBB_NUM_THREADS=1
echo "Doing 1 Thread on $(date +"%F-%H-%M-%S")"
for ((i=0; i < $3; i++));
do
	./$1.$2
done

for((j=8; j <= 265; j+=8))
do
	export TBB_NUM_THREADS=$j
	echo "Doing $j Threads on $(date +"%F-%H-%M-%S")"

	for ((i=0; i < $3; i++));
	do
		./$1.$2
	done
done
