# Patches for the relu_stable code

The patches are based on the `af53612` commit of
[relu_stable](https://github.com/MadryLab/relu_stable). The training and
verification results should be placed in [../data](../data).

## Training for MNIST

```bash
cd /path/to/relu_stable
git checkout af53612
patch -Np1 -i /path/to/relu_stable_patch/mnist.patch
python train.py
./verification/verify.sh example 0.1
```

## Training for CIFAR10

```bash
cd /path/to/relu_stable
git checkout af53612
patch -Np1 -i /path/to/relu_stable_patch/cifar10.patch
python train_naive_cifar_ia.py
./verification/verify.sh example 0.00784313725490196
```
