# DDP-training-example
An example of DDP(Distributed Data Parallel) type multi-gpu training. Based on https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html?highlight=parameter

## Usage
Train from scratch. Using CUDA_VISIBLE_DEVICES to select which gpus to use.
```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --standalone --nnodes=1 --nproc-per-node=2 YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```