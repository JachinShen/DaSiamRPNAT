#!/bin/sh
for line in `cat SEQUENCES`
do
    unzip $line.zip -d ./OTB/
done

