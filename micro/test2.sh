#!/bin/sh

NOW=$(date +"%F-%H-%M-%S")
FILENAME=C++STL_micro_$NOW.txt

echo -e "Did $1 runs of microbenchmark on $NOW using the C++ STL version\n-------------------------------------------------------" >> $FILENAME

for((j=2; j <= 24; j+=2))
do
	export TBB_NUM_THREADS=$j
	echo "Doing $j Threads"
	
	for ((i=0; i < $1; i++));
	do
		./micro >> $FILENAME
	done

done

scp $FILENAME mietzsch@remote.cip.ifi.lmu.de:/home/m/mietzsch/BA_Test/$FILENAME
