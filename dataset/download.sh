#!/bin/sh
for line in `cat SEQUENCES`
do
    wget http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/$line.zip
done

