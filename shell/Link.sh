#!/bin/sh

cd ..


ls *.txt | xargs -I{} ln -s {} link.{}
