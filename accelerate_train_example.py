import argparse
import os
import logging
from tqdm import trange

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from accelerate import Accelerator

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
# Train options
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size (per gpu)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='the learning rate')

# Log options
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='directory to store results')
parser.add_argument('--save_freq', type=int, default=5, help='number of epochs between saving training results')

accelerator = Accelerator()

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
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        accelerator.backward(loss)
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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    logging.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

if __name__ == '__main__':
    args = parser.parse_args()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(train_dataloader, test_dataloader, model, optimizer)

    for epoch in trange(args.epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer, args)
        test_loop(test_dataloader, model, loss_fn, args)

        if ((epoch + 1) % args.save_freq) == 0 and accelerator.is_main_process:
            os.makedirs(args.results_dir, exist_ok=True)
            save_file = os.path.join(args.results_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_file)
            logging.info(f"Checkpoint has been saved as {save_file}")

    print("Done!")