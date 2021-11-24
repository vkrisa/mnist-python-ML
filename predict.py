import torch
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import numpy as np
import torch.nn.functional as F
from model import MyNet

img_loc = "1.png"
model = MyNet().to("cuda")
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

state_dict = torch.load('data/weights')
model.load_state_dict(state_dict)
im = Image.open(img_loc).convert('L')
im = im.resize((28, 28))
im.show()

data = np.array(im)
data = 255-data
data = data.reshape(1, 28, 28)
data = torch.tensor(data, dtype=torch.float)

with torch.no_grad():
    d = data.view(-1, 1, 28, 28).to('cuda')
    y_pred = model(d)

    m = F.softmax(y_pred, dim=1)
    y_pred_value = torch.argmax(m, 1)



