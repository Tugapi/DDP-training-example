import argparse
import os
import logging
from tqdm import trange

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
# Train options
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size (per gpu)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='the learning rate')

# Log options
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='directory to store results')
parser.add_argument('--print_freq', type=int, default=5, help='number of epochs between printing training results')
parser.add_argument('--save_freq', type=int, default=5, help='number of epochs between saving training results')

# DDP options
parser.add_argument('--local_rank', typr=int, default=-1, help='local device id on current node')

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, args):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda(args.local_rank)
        y = y.cuda(args.local_rank)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn, args):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda(args.local_rank)
            y = y.cuda(args.local_rank)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    logging.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = Model().to(args.loacl_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    loss_fn = nn.CrossEntropyLoss().to(args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    for epoch in trange(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loop(train_dataloader, model, loss_fn, optimizer, args)
        test_loop(test_dataloader, model, loss_fn, args)
        if ((epoch + 1) % args.print_freq) == 0:
            loss, current = loss_fn.item(), (epoch + 1)
            logging.info(f"Training loss: {loss:>7f}  [current epoch: {current:>5d}]")
        if ((epoch + 1) % args.save_freq) == 0 and torch.distributed.get_rank() == 0:
            os.makedirs(args.results_dir, exist_ok=True)
            save_file = os.path.join(args.results_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_file)
            logging.warning(f"Checkpoint has been saved as {save_file}")

    print("Done!")

## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py