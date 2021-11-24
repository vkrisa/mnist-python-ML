import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


df = pd.read_csv('data/train.csv')

labels = df["label"].to_numpy()
data = df.drop("label", 1).to_numpy()
data = data.reshape(-1, 28, 28)

for i in range(len(data)):
    folder = 'data/{}'.format(labels[i])

    if not os.path.exists(folder):
        os.mkdir(folder)

    img = Image.fromarray(np.uint8(data[i]), 'L')
    img.save(folder+'/{}.png'.format(i))







