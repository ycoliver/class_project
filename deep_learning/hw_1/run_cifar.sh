
### base experiment

python cifar10_experiments.py --exp_name base_experiment
python cifar10_experiments.py --is_large_hidden --exp_name add_hidden_dim
python cifar10_experiments.py --is_large_layer --exp_name add_layer_num
python cifar10_experiments.py --mean_pooling --exp_name use_mean_pooling
python cifar10_experiments.py --is_resnet --exp_name use_resnet
python cifar10_experiments.py --is_l2_loss --exp_name use_l2_regular
python cifar10_experiments.py --use_adam --exp_name use_adam