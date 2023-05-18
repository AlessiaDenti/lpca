## CIFAR-10 Clean Set ##############################

# Baseline (cross-entropy loss)
python train_preactresnet.py --train-dir ./data/CIFAR10/train/ --val-dir ./data/CIFAR10/val/ --dataset CIFAR10 --arch PreActResNet18 --out-dir ./checkpoints/cifar10_preActResNet18 --gpu 0

# Nested=10
python train_preactresnet.py --train-dir ./data/CIFAR10/train/ --val-dir ./data/CIFAR10/val/ --dataset CIFAR10 --arch PreActResNet18 --out-dir ./checkpoints/cifar10_preActResNet18_nbpc100_nested10 --num-pc 100 --nested 10 --gpu 0


## CIFAR-10 Symmetric 20% ##############################

# Nested=10
python train_preactresnet.py --train-dir ./data/CIFAR10/train_sn_0.2/ --val-dir ./data/CIFAR10/val/ --dataset CIFAR10 --arch PreActResNet18 --out-dir ./checkpoints/cifar10sn0.2_preActResNet18_nbpc100_nested10 --num-pc 100 --nested 10 --gpu 0




## for testing ##############################
python test_preactresnet.py test.py --test-dir ./data/CIFAR10/test/ --dataset CIFAR10 --arch PreActResNet18 --resumePthList ./checkpoints/cifar10sn0.2_preActResNet18_nbpc100_nested10 --KList 10 --gpu 0
