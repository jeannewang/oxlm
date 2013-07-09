#!/bin/sh
train_sgd --input wsj2-21.words.co6 --word-width 100 -n 5 --iterations 100 --step-size 0.05 --minibatch 10000 --threads 1 --test-set wsj22.words.co6 --label-sample-size 25 --lambda 1 --randomise --diagonal
