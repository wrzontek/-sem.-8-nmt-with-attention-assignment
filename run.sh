#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./en_pl_data/train.en --train-tgt=./en_pl_data/train.pl --dev-src=./en_pl_data/dev.en --dev-tgt=./en_pl_data/dev.pl --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./en_pl_data/test.en ./en_pl_data/test.pl outputs/test_outputs.txt --cuda
elif [ "$1" = "test_full" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./en_pl_data/test.en ./en_pl_data/test.pl outputs/test_outputs.txt --cuda --any_unks
elif [ "$1" = "train_local" ]; then
	python3 run.py train --train-src=./en_pl_data/train.en --train-tgt=./en_pl_data/train.pl --dev-src=./en_pl_data/dev.en --dev-tgt=./en_pl_data/dev.pl --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python3 run.py decode model.bin ./en_pl_data/test.en ./en_pl_data/test.pl outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./en_pl_data/train.en --train-tgt=./en_pl_data/train.pl vocab.json
else
	echo "Invalid Option Selected"
fi
