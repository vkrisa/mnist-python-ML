import pathlib
import os
import numpy as np
import shutil

root = pathlib.Path('data')
for entry in root.iterdir():
    if entry.is_dir():
        files = os.listdir(entry)
        files = list(filter(lambda x: x.endswith(".png"), files))
        test = np.random.choice(files, len(files) // 10)
        train = set(files) - set(test)
        test = set(files) - set(train)
        train_dir = pathlib.Path(entry) / 'train'
        test_dir = pathlib.Path(entry) / 'test'

        if not train_dir.is_dir():
            os.mkdir(train_dir)
            for file in train:
                shutil.move(os.path.join(entry, file), train_dir / file)

        if not test_dir.is_dir():
            os.mkdir(test_dir)
            for file in test:
                shutil.move(os.path.join(entry, file), test_dir / file)