#!/bin/bash

cd ..
echo timingcut.sh inputdir outputdir

inputdir=$1
outputdir=$2

mkdir $outputdir
for file in `ls $inputdir`; do
    echo Processing $file ...
    python timingcut.py -i $inputdir/$file -o $outputdir/$file
done

