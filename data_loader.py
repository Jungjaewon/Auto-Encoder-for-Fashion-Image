from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import json
import numpy as np
import pickle as cp


class Polyvore(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, image_dir, train_json, test_json, transform, mode, train_pickle, test_pickle):
        """Initialize and preprocess the Polyevore dataset."""
        self.image_dir = image_dir
        self.train_json = train_json
        self.test_json = test_json
        self.transform = transform
        self.mode = mode
        self.dataset = list()
        self.train_pickle = train_pickle
        self.test_pickle = test_pickle
        random.seed(2344)
        self.preprocess()

    def preprocess(self):

        if self.mode == 'train' and os.path.exists(self.train_pickle):
            with open(self.train_pickle,'rb') as fp:
                print('training list is restored from pickle file')
                self.dataset = cp.load(fp)
                random.shuffle(self.dataset)
                random.shuffle(self.dataset)
                return
        elif self.mode == 'test' and os.path.exists(self.test_pickle):
            with open(self.test_pickle,'rb') as fp:
                print('test list is restored from pickle file')
                self.dataset = cp.load(fp)
                random.shuffle(self.dataset)
                random.shuffle(self.dataset)
                return

        dataset_json = self.train_json if self.mode == 'train' else self.test_json

        with open(dataset_json,'r') as fp:
            json_array = json.load(fp)

        for outfit in json_array:
            set_id = outfit['set_id']
            outfits = outfit['items']

            for i in range(1, len(outfits)):
                self.dataset.append([set_id,i])

            random.shuffle(self.dataset)

        random.shuffle(self.dataset)
        random.shuffle(self.dataset)

        if self.mode == 'train':
            data_list_filename = './train_datalist.pickle'
        else:
            data_list_filename = './test_datalist.pickle'

        with open(data_list_filename,'wb') as fp:
            cp.dump(self.dataset,fp)


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        set_id, image_n = self.dataset[index]
        #print '__get_item',set_id, origin_num, origin_label, dest_num, dest_label

        target_image = Image.open(os.path.join(self.image_dir, str(set_id), str(image_n) + '.jpg'))
        target_image = target_image.convert('RGB')

        return torch.Tensor([int(set_id)]), self.transform(target_image)

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)


def get_loader(image_dir, train_json, test_json, image_size=256,
               batch_size=16, mode='train', num_workers=1, train_pickle = '', test_pickle = ''):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())

    transform.append(T.Resize((image_size,image_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = Polyvore(image_dir, train_json, test_json, transform, mode, train_pickle, test_pickle)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
