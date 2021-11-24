import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torchvision


class MnistDataset(Dataset):
    def __init__(self, root_dir, directory):
        self.dir = directory
        self._root = pathlib.Path(root_dir)
        self._files = tuple(self._traverse(self._root, directory))
        self._data = self._np_data(self._files)
        self.trans = torchvision.transforms.RandomApply([
            torchvision.transforms.RandomAffine(5),
            torchvision.transforms.RandomPerspective(p=0.5),
            torchvision.transforms.RandomRotation([-5, 5], resample=Image.BICUBIC),
            torchvision.transforms.RandomResizedCrop(size=28, scale=(0.8, 1))
        ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        label, image_path = self._data[item]
        image = Image.open(image_path)
        if self.dir == 'train':
            image = self.trans(image)
        image_array = np.asarray(image)
        image_array = image_array.reshape(1, 28, 28)
        return torch.tensor(image_array, dtype=torch.float), self._label_array(label)

    def _traverse(self, root: pathlib.Path, directory: str):
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix == '.png':
                if directory in str(entry).split('/')[2]:
                    yield str(entry)
            elif entry.is_dir():
                yield from self._traverse(entry, directory)

    def _np_data(self, files):
        number = tuple(map(lambda x: x.split('/')[1], files))
        df = pd.DataFrame({"number": number, "path": files})
        vector = df.to_numpy()
        return vector

    def _label_array(self, label):
        empty_tensor = torch.tensor(int(label)).long()
        return empty_tensor


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64*7*7, 100),
            nn.ReLU()
        )
        self.out = nn.Linear(100, 10)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.out(out)

        return out


if __name__ == "__main__":
    device = 'cuda:0'
    train = MnistDataset('data', 'train')
    test = MnistDataset('data', 'test')
    model = MyNet().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train, batch_size=24, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=24, shuffle=True)

    scores = []
    losses = []

    for epoch in range(20):
        for x, y in tqdm(train_dataloader):
            train = x.view(-1, 1, 28, 28).to(device)
            label = y.to(device)

            y_pred = model(train)

            loss = loss_fn(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        score = 0

        for x, y in tqdm(test_dataloader):
            train = x.view(-1, 1, 28, 28).to(device)
            label = y.to(device)

            y_pred = model(train)

            m = F.softmax(y_pred, dim=1)
            y_pred_value = torch.argmax(m, 1)

            score += (y_pred_value == label).sum().item()

        scores.append(score/len(test))
        losses.append(loss.item())

        print("epoch "+str(epoch+1) + ":" + str(score / len(test)))

        if epoch==10:
            learning_rate = 1e-5

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(scores)
    ax1.set_title("scores")
    ax2.plot(losses)
    ax2.set_title("losses")
    plt.show()

    torch.save(model.state_dict(), 'data/weights')















