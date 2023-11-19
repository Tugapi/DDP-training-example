import torch
from torch import nn, optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)  # get local_rank from arguments
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP benchmark initialize
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')

device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

inputs = torch.randn(20, 10).to(local_rank)
outputs = model(inputs)
labels = torch.ones(20, 10).to(local_rank)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss_fn(outputs, labels).backward()
optimizer.step()
optimizer.zero_grad()

# Bash commandlines: python -m torch.distributed.launch --nproc_per_node 4 (number of gpus) after.py