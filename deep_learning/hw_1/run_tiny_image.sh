
### base experiment
export PYTHONPATH=/Users/daixunlian/workspace/class_project/deep_learningn:$PYTHONPATH
python tiny_imagenet_experiment.py --exp_name base_experiment
python tiny_imagenet_experiment.py --hidden_dim 2 --exp_name add_hidden_dim
python tiny_imagenet_experiment.py --layer_num 6 --exp_name add_layer_num
python tiny_imagenet_experiment.py --mean_pooling --exp_name use_mean_pooling
python tiny_imagenet_experiment.py --is_resnet --exp_name use_resnet
python tiny_imagenet_experiment.py --is_l2_loss --exp_name use_l2_regular
python tiny_imagenet_experiment.py --use_adam --exp_name use_adam