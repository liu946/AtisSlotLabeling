#!/usr/bin/env bash

for word in 200; do
    for hidden in 150; do
        echo -----------
        echo ${word} ${hidden}
        echo -----------
        python ./main.py train ${word} ${hidden}
        ./eval/conlleval.pl -d "\t" < pred-${word}-${hidden}.out
    done
done
