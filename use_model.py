import torch
import matplotlib.pyplot as plt
from torch import nn
from model import MyNet

model = MyNet().to("cuda")


state_dict = torch.load('data/weights')
model.load_state_dict(state_dict)

w1 = model.layer1[0].weight.detach().cpu().numpy()
for i in range(len(w1)):
    plt.figure()
    plt.imshow(w1[i][0])
    plt.show()
