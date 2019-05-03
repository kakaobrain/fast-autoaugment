# Fast AutoAugment

Official [Fast AutoAugment](https://arxiv.org/abs/1905.00397) Implementation in PyTorch.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

We do not open augmentation search codes at this moment, but it will be publicly open with our follow-up studies.

## Results

### Cifar-10 / 100

Search : **3.5 GPU Hours (1428x faster than AutoAugment)**, WResnet40x2 on Reduced Cifar10

| Model(Cifar10)          | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-------------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2        | 5.3        | 4.1        | 3.7         | 3.6 / 3.7        |
| Wide-ResNet-28-10       | 3.9        | 3.1        | 2.6         | 2.7 / 2.7        |
| Shake-Shake(26 2x32d)   | 3.6        | 3.0        | 2.5         | 2.7 / 2.5        |
| Shake-Shake(26 2x96d)   | 2.9        | 2.6        | 2.0         | 2.0 / 2.0        |
| Shake-Shake(26 2x112d)  | 2.8        | 2.6        | 1.9         | 2.0 / 1.9        |
| PyramidNet+ShakeDrop    | 2.7        | 2.3        | 1.5         | 1.8 / 1.7        |

| Model(Cifar100)       | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-----------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2      | 26.0       | 25.2       | 20.7        | 20.7 / 20.6      |
| Wide-ResNet-28-10     | 18.8       | 28.4       | 17.1        | 17.8 / 17.5      |
| Shake-Shake(26 2x96d) | 17.1       | 16.0       | 14.3        | 14.9 / 14.6      |
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.9 / 11.7      |

### Imagenet

Search : **450 GPU Hours (33x faster than AutoAugment)**, Resnet50 on Reduced Imagenet

| Model      | Baseline   | AutoAugment | Fast AutoAugment |
|------------|------------|-------------|------------------|
| Resnet-50  | 23.7 / 6.9 | 22.4 / 6.2  | **21.4 / 5.9**   |
| Resnet-200 | 21.5 / 5.8 | 20.0 / 5.0  | **19.4 / 4.7**   |


## Run

You can train network architectures on cifar10/100 and imagenet with our searched policies.

- fa_reduced_cifar10 : reduced cifar10(4k images), wresnet40x2
- fa_reduced_imagenet : reduced imagenet(50k images, 120 classes), resnet50

```
$ python train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
$ python train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
```

Note that we conducted experiments with imagenet dataset using 8 machines with four V100 gpus each.

```
$ python train.py -c confs/resnet50_imagenet_b4096.yaml --aug fa_reduced_imagenet --horovod
```

## Citation

If you use any part of this code in your research, please cite our [paper](https://arxiv.org/abs/1905.00397).

```
@article{lim2019fast,
  title={Fast AutoAugment},
  author={Lim, Sungbin and Kim, Ildoo and Kim, Taesup and Kim, Chiheon and Kim, Sungwoong},
  journal={arXiv preprint arXiv:1905.00397},
  year={2019}
}
```

## Contact for Issues
- Ildoo Kim, ildoo.kim@kakaobrain.com
- Sungbin Lim, sungbin.lim@kakaobrain.com


## References & Opensources

1. ResNet References
    - (ResNet) Deep Residual Learning for Image Recognition
      - Paper : https://arxiv.org/abs/1512.03385
    - (ResNet) Identity Mappings in Deep Residual Networks
      - Paper : https://arxiv.org/abs/1603.05027
    - Codes
      - https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv/models
2. (PyramidNet) Deep Pyramidal Residual Networks
    - Paper : https://arxiv.org/abs/1610.02915
    - Author's Code : https://github.com/dyhan0920/PyramidNet-PyTorch
3. (Wide-ResNet)
    - Code : https://github.com/meliketoy/wide-resnet.pytorch
4. (Shake-Shake)
    - Code : https://github.com/owruby/shake-shake_pytorch
5. ShakeDrop Regularization for Deep Residual Learning
    - Paper : https://arxiv.org/abs/1802.02375
    - Code : https://github.com/owruby/shake-drop_pytorch
6. LARS : Large Batch Training of Convolutional Networks
    - Paper : https://arxiv.org/abs/1708.03888
    - Code : https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
7. (ARS-Aug) Learning data augmentation policies using augmented random search
    - Paper : https://arxiv.org/abs/1811.04768
    - Author's Code : https://github.com/gmy2013/ARS-Aug
8. AutoAugment
    - Code : https://github.com/tensorflow/models/tree/master/research/autoaugment
9. https://pytorch.org/docs/stable/torchvision/models.html
10. https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
11. Ray
    - https://github.com/ray-project/ray
    - https://ray.readthedocs.io/en/latest/tune.html
    - https://medium.com/formcept/scaling-python-modules-using-ray-framework-e5fc5430dc3e
12. HyperOpt
    - https://github.com/hyperopt/hyperopt
