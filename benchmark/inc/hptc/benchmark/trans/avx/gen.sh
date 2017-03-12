#!/bin/bash

NUM_PATTERN='^[0-9]+$'

if ! [[ $1 =~ $NUM_PATTERN ]]; then
  echo "Usage: bash gen.sh [NUM_OF_THREADS]"
  exit 1
fi

if [ $1 -le 1 ]; then
  GEN_THREAD_NUM=2
else
  GEN_THREAD_NUM=$1
fi

export KMP_AFFINITY=compact,1
export OMP_NUM_THREADS=$GEN_THREAD_NUM
ttc --perm=1,0 --size=7248,7248 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0 --size=43408,1216 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0 --size=1216,43408 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,2,1 --size=368,384,384 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,2,1 --size=2144,64,384 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,2,1 --size=368,64,2307 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,2 --size=384,384,355 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,2 --size=2320,384,59 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,2 --size=384,2320,59 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,0 --size=384,355,384 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,0 --size=2320,59,384 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,0 --size=384,59,2320 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,1 --size=80,96,75,96 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,1 --size=464,16,75,96 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,1 --size=80,16,75,582 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,3,0 --size=96,75,96,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,3,0 --size=608,12,96,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,1,3,0 --size=96,12,608,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,3,1 --size=96,75,96,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,3,1 --size=608,12,96,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,3,1 --size=96,12,608,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,3,2 --size=96,96,75,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,3,2 --size=608,96,12,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,0,3,2 --size=96,608,12,75 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,0 --size=96,75,75,96 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,0 --size=608,12,75,96 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,0 --size=96,12,75,608 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,4,2,1,3 --size=32,48,28,28,48 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,4,2,1,3 --size=176,8,28,28,48 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,4,2,1,3 --size=32,8,28,28,298 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,4,0 --size=48,28,28,48,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,4,0 --size=352,4,28,48,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,1,4,0 --size=48,4,28,352,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,3 --size=48,28,48,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,3 --size=352,4,48,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,3 --size=48,4,352,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,3,0,4,2 --size=48,48,28,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,3,0,4,2 --size=352,48,4,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=1,3,0,4,2 --size=48,352,4,28,28 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=4,3,2,1,0 --size=48,28,28,28,48 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=4,3,2,1,0 --size=352,4,28,28,48 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=4,3,2,1,0 --size=48,4,28,28,352 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,5,4,1 --size=16,32,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,5,4,1 --size=48,10,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=0,3,2,5,4,1 --size=16,10,15,103,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,0,5,1,4 --size=32,15,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,0,5,1,4 --size=112,5,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,0,5,1,4 --size=32,5,15,112,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,5,3 --size=32,15,32,15,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,5,3 --size=112,5,32,15,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=2,0,4,1,5,3 --size=32,5,112,15,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,5,1,0,4 --size=32,15,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,5,1,0,4 --size=112,5,15,32,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=3,2,5,1,0,4 --size=32,5,15,112,15,15 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=5,4,3,2,1,0 --size=32,15,15,15,15,32 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=5,4,3,2,1,0 --size=112,5,15,15,15,32 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM
ttc --perm=5,4,3,2,1,0 --size=32,5,15,15,15,112 --alpha=2.3 --beta=4.2 --compiler=icpc --ignoreDatabase --numThreads=$GEN_THREAD_NUM

rm log.txt
mv ./ttc_transpositions ./ttc_implementations
