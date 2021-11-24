import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

epoch = 3


class MnistDataset(Dataset):
    def __init__(self, root_dir, directory):
        self._root = pathlib.Path(root_dir)
        self._files = tuple(self._traverse(self._root, directory))
        self._data = self._np_data(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        label, image_path = self._data[item]
        image = Image.open(image_path)
        image_array = np.asarray(image)
        flat_image = image_array.flatten()
        return torch.tensor(flat_image, dtype=torch.float), self._label_array(label)

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
    def __init__(self, d_in, hidden, d_out):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden)
        self.linear2 = torch.nn.Linear(hidden, 10)
        self.linear3 = torch.nn.Linear(10, d_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        y_pred = self.linear3(x)
        return y_pred


if __name__ == "__main__":
    device = 'cuda:0'
    train = MnistDataset('data', 'train')
    test = MnistDataset('data', 'test')

    model = MyNet(784, 100, 10).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True)

    scores = []
    losses = []

    for i in range(epoch):
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        score = 0

        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            m = F.softmax(y_pred, dim=1)
            y_pred_value = torch.argmax(m, 1)

            score += (y_pred_value == y).sum().item()

        scores.append(score/len(test))
        losses.append(loss.item())

        print(score / len(test))

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















