#!/bin/bash
for dir in *
do
    cd $dir
    mv * $dir\.uniref
    cd ..
done
