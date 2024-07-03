from config import *
from easydl import *
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

class BaseImageDataset(Dataset):
    """
    Base image dataset for handling images stored in .npy files.

    Subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x: x)
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        im = np.load(self.datas[index])  # Load .npy file
        im = Image.fromarray(im)  # Convert numpy array to PIL Image for compatibility with transforms
        im = self.transform(im)
        if not self.return_id:
            return im, self.labels[index]
        return im, self.labels[index], index

    def __len__(self):
        return len(self.datas)


class FileListDataset(BaseImageDataset):
    """
    Dataset that consists of a file which has the structure of:

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id.
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id=return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x: True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line:  # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [os.path.join(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('Invalid label number, maybe there is a space in the image path?')
                raise e

        self.num_classes = num_classes or max(self.labels) + 1

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''

source_private_classes = [i for i in range(4)]
target_private_classes = [4]
common_classes = source_private_classes

source_classes = source_private_classes
target_classes = target_private_classes

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefix,
                            transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefix,
                            transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefix,
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefix,
                            transform=test_transform, filter=(lambda x: x in target_classes))

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
