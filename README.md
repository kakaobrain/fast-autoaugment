# Fast AutoAugment **(Accepted at NeurIPS 2019)**

Official [Fast AutoAugment](https://arxiv.org/abs/1905.00397) implementation in PyTorch.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

<p align="center">
<img src="etc/search.jpg" height=350>
</p>

## Results

### CIFAR-10 / 100

Search : **3.5 GPU Hours (1428x faster than AutoAugment)**, WResNet-40x2 on Reduced CIFAR-10

| Model(CIFAR-10)         | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |   |
|-------------------------|------------|------------|-------------|------------------|----|
| Wide-ResNet-40-2        | 5.3        | 4.1        | 3.7         | 3.6 / 3.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_wresnet40x2_top1_3.52.pth) |
| Wide-ResNet-28-10       | 3.9        | 3.1        | 2.6         | 2.7 / 2.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_wresnet28x10_top1.pth) |
| Shake-Shake(26 2x32d)   | 3.6        | 3.0        | 2.5         | 2.7 / 2.5        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x32d_top1_2.68.pth) |
| Shake-Shake(26 2x96d)   | 2.9        | 2.6        | 2.0         | 2.0 / 2.0        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x96d_top1_1.97.pth) |
| Shake-Shake(26 2x112d)  | 2.8        | 2.6        | 1.9         | 2.0 / 1.9        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x112d_top1_2.04.pth) |
| PyramidNet+ShakeDrop    | 2.7        | 2.3        | 1.5         | 1.8 / 1.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_pyramid272_top1_1.44.pth) |

| Model(CIFAR-100)      | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |    |
|-----------------------|------------|------------|-------------|------------------|----|
| Wide-ResNet-40-2      | 26.0       | 25.2       | 20.7        | 20.7 / 20.6      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_wresnet40x2_top1_20.43.pth) |
| Wide-ResNet-28-10     | 18.8       | 18.4       | 17.1        | 17.3 / 17.3      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_wresnet28x10_top1_17.17.pth) |
| Shake-Shake(26 2x96d) | 17.1       | 16.0       | 14.3        | 14.9 / 14.6      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_shake26_2x96d_top1_15.15.pth) |
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.9 / 11.7      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_pyramid272_top1_11.74.pth) |

### ImageNet

Search : **450 GPU Hours (33x faster than AutoAugment)**, ResNet-50 on Reduced ImageNet

| Model      | Baseline   | AutoAugment | Fast AutoAugment<br/>(Top1/Top5) |    |
|------------|------------|-------------|------------------|----|
| ResNet-50  | 23.7 / 6.9 | 22.4 / 6.2  | **22.4 / 6.3**   | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/imagenet_resnet50_top1_22.2.pth) |
| ResNet-200 | 21.5 / 5.8 | 20.0 / 5.0  | **19.4 / 4.7**   | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/imagenet_resnet200_top1_19.4.pth) |

Notes
* We evaluated resnet-50 and resnet-200 with resolution of 224 and 320, respectively. According to the original resnet paper, resnet 200 was tested with the resolution of 320. Also our resnet-200 baseline's performance was similar when we use the resolution.
* But with recent our code clean-up and bugfixes, we've found that the baseline performs similar to the baseline even using 224x224.
* When we use 224x224, resnet-200 performs **20.0 / 5.2**. Download link for the trained model is [here](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/imagenet_resnet200_res224.pth).

We have conducted additional experiments with EfficientNet.

| Model | Baseline   | AutoAugment |   | Our Baseline(Batch) | +Fast AA |
|-------|------------|-------------|---|---------------------|----------|
| B0    | 23.2       | 22.7        |   | 22.96               | 22.68    |

### SVHN Test

Search : **1.5 GPU Hours**

|                                  | Baseline | AutoAug / Our | Fast AutoAugment  |
|----------------------------------|---------:|--------------:|--------:|
| Wide-Resnet28x10                 | 1.5      | 1.1           | 1.1     |

## Run

We conducted experiments under

- python 3.6.9
- pytorch 1.2.0, torchvision 0.4.0, cuda10

### Search a augmentation policy

Please read ray's document to construct a proper ray cluster : https://github.com/ray-project/ray, and run search.py with the master's redis address.

```
$ python search.py -c confs/wresnet40x2_cifar10_b512.yaml --dataroot ... --redis ...
```

### Train a model with found policies

You can train network architectures on CIFAR-10 / 100 and ImageNet with our searched policies.

- fa_reduced_cifar10 : reduced CIFAR-10(4k images), WResNet-40x2
- fa_reduced_imagenet : reduced ImageNet(50k images, 120 classes), ResNet-50

```
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python FastAutoAugment/train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python FastAutoAugment/train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
$ python FastAutoAugment/train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python FastAutoAugment/train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
...
$ python FastAutoAugment/train.py -c confs/resnet50_b512.yaml --aug fa_reduced_imagenet
$ python FastAutoAugment/train.py -c confs/resnet200_b512.yaml --aug fa_reduced_imagenet
```

By adding --only-eval and --save arguments, you can test trained models without training.

If you want to train with multi-gpu/node, use `torch.distributed.launch` such as

```bash
$ python -m torch.distributed.launch --nproc_per_node={num_gpu_per_node} --nnodes={num_node} --master_addr={master} --master_port={master_port} --node_rank={0,1,2,...,num_node} FastAutoAugment/train.py -c confs/efficientnet_b4.yaml --aug fa_reduced_imagenet
```

## Citation

If you use this code in your research, please cite our [paper](https://arxiv.org/abs/1905.00397).

```
@inproceedings{lim2019fast,
  title={Fast AutoAugment},
  author={Lim, Sungbin and Kim, Ildoo and Kim, Taesup and Kim, Chiheon and Kim, Sungwoong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## Contact for Issues
- Ildoo Kim, ildoo.kim@kakaobrain.com

## References & Opensources

We increase the batch size and adapt the learning rate accordingly to boost the training. Otherwise, we set other hyperparameters equal to AutoAugment if possible. For the unknown hyperparameters, we follow values from the original references or we tune them to match baseline performances.

- **ResNet** : [paper1](https://arxiv.org/abs/1512.03385), [paper2](https://arxiv.org/abs/1603.05027), [code](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv/models)
- **PyramidNet** : [paper](https://arxiv.org/abs/1610.02915), [code](https://github.com/dyhan0920/PyramidNet-PyTorch)
- **Wide-ResNet** : [code](https://github.com/meliketoy/wide-resnet.pytorch)
- **Shake-Shake** : [code](https://github.com/owruby/shake-shake_pytorch)
- **ShakeDrop Regularization** : [paper](https://arxiv.org/abs/1802.02375), [code](https://github.com/owruby/shake-drop_pytorch)
- **AutoAugment** : [code](https://github.com/tensorflow/models/tree/master/research/autoaugment)
- **Ray** : [code](https://github.com/ray-project/ray)
- **HyperOpt** : [code](https://github.com/hyperopt/hyperopt)
