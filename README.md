# Multi-GPU training example
An example of DDP(Distributed Data Parallel) type multi-gpu training. Based on https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html?highlight=parameter
Additionally integrate Accelerate into the training code. 
## Usage
### DDP-type training
Train from scratch. Using CUDA_VISIBLE_DEVICES to select which gpus to use.
```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --standalone --nnodes=1 --nproc-per-node=2 YOUR_TRAINING_SCRIPT.py(DDP_train_example.py) (--arg1 ... train script args...)
```
### accelerate-type training 
First run
```bash
accelerate config
```
answer the questions asked. This will generate a config file that will be used automatically to properly set the default options when doing
```bash
accelerate launch YOUR_TRAINING_SCRIPT.py(DDP_train_example.py) (--arg1 ... train script args...)
```
You can also directly pass in the arguments you would to `torchrun` as arguments to `accelerate launch` if you wish to not run `accelerate config`.
For example,
```bash
accelerate launch --multi_gpu --num_processes 2 YOUR_TRAINING_SCRIPT.py(DDP_train_example.py) (--arg1 ... train script args...)
```