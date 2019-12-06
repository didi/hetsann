
python -u execute_sparse.py --seed 0 --dataset imdb --target_node 0 --target_is_multilabels 1 --train_rate 0.8 --hid 8 --heads 8 --lr 1e-3 --epochs 1000 --patience 30 --layers 3 --residue --loop_coef 1e-3 --inv_coef 1e-5  
