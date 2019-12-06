#!/bin/bash
python -u execute_sparse.py --seed 0 --dataset aminer --train_rate 0.8 --hid 8 --heads 8 --epochs 1000 --patience 30 --target_node 0 1 --target_is_multilabels 0 1 --layers 3
