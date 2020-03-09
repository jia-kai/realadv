# realadv

Code for the paper [Exploiting Verified Neural Networks via Floating Point
Numerical Error](https://arxiv.org/abs/2003.03021).

## Steps for reproduction

1. Prepare the requirements: python >= 3.8, numpy, pytorch, cython and opencv.
   Julia and [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) are also
   required. You may need my [fork](https://github.com/jia-kai/MIPVerify.jl) of
   MIPVerify unless the [pull request](https://github.com/vtjeng/MIPVerify.jl/pull/34)
   is merged.
2. Train the MNIST and CIFAR10 models and get the verification results following
   the instructions given in
   [relu_stable](https://github.com/MadryLab/relu_stable). Note that the
   original repo only contains an MNIST model. You can apply the patches in
   [relu_stable_patch](relu_stable_patch) to reproduce the training step.
   I have also included pre-trained model weights and verification results in
   [data](data) so this step can be skipped.
3. Run the scripts [`step0_find_edge_input.sh`](step0_find_edge_input.sh),
   [`step1_find_edge_model.sh`](step1_find_edge_model.sh) and
   [`step2_attack.sh`](step2_attack.sh) or
   [`attack_parallel.sh`](attack_parallel.sh) to reproduce the results. Please
   read the scripts to get a basic understanding of what they are doing.

Attack logs and adversarial images for the experiments reported in the paper are
available in [result](result). Run `python -m realadv view_attack` to view
adversarial images.

## Citation

```txt
@misc{jia2020exploiting,
    title={Exploiting Verified Neural Networks via Floating Point Numerical Error},
    author={Kai Jia and Martin Rinard},
    year={2020},
    eprint={2003.03021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
