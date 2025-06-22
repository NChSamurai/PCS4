#!/bin/bash
module load cuda/11.4

export SIZE=500000

echo "Nods" | tee -a ~/Desktop/Lab2/outputrun1.log
for i in {1..30}
do
        ./progCUDA4
        sync
done
./GetMiddleTime
echo "Array size: $SIZE" | tee -a ~/Desktop/Lab2/outputrun1.log

rm GeneralTime.bin

export SIZE=5000000

echo "Nods" | tee -a ~/Desktop/Lab2/outputrun1.log
for i in {1..30}
do
        ./progCUDA4
        sync
done
./GetMiddleTime
echo "Array size: $SIZE" | tee -a ~/Desktop/Lab2/outputrun1.log

rm GeneralTime.bin

export SIZE=10000000

echo "Nods" | tee -a ~/Desktop/Lab2/outputrun1.log
for i in {1..30}
do
        ./progCUDA4
        sync
done
./GetMiddleTime
echo "Array size: $SIZE" | tee -a ~/Desktop/Lab2/outputrun1.log

rm GeneralTime.bin
