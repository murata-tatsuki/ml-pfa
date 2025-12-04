#!/bin/bash

cd ..
echo eventcut.sh inputdir outputdir

inputdir=$1
outputdir=$2

mkdir $outputdir
for file in `ls $inputdir`; do
    echo Processing $file ...
    python eventcut.py -i $inputdir/$file -o $outputdir/$file
done

