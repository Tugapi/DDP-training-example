import torch
from torch import nn, optim

model = nn.Linear(10, 10)

inputs = torch.randn(20, 10)
outputs = model(inputs)
labels = torch.ones(20, 10)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss_fn(outputs, labels).backward()
optimizer.step()
optimizer.zero_grad()