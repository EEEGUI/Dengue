import torch
from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.Adam([], lr=0.1)

scheduler = StepLR(optimizer, 10, 0.1)
for epoch in range(30):
    for i in range(100):
        scheduler.step()
        print('epoch:%d, step:%d, learning_rate:%.5f' % (epoch, i, scheduler.get_lr()[0]))
