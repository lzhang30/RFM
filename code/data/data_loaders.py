import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import read_list, read_data, softmax
from utils.config import Config


class Synapse_AMOS(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False, is_val=False, task="synapse", num_cls=1):
        self.ids_list = read_list(split, task=task)
        self.config = Config(task)
        self.split = split
  
        self.repeat = repeat
        self.task=task
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        
        if task == 'word':
            self.num_cls = 17
        
        self.num_cls = num_cls
        self._weight = None
        self.is_val = is_val
        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list): # <-- load data to memory
                image, label = read_data(data_id, task=task)
                self.data_list[data_id] = (image, label)


    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        # [160, 384, 384]
        #print(data_id)
        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            image, label = read_data(data_id, task=self.task,unlabeled = self.unlabeled)
        return data_id, image, label


    def __getitem__(self, index):
        
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        #print(data_id)
        _, image, label = self._get_data(data_id)
        if self.unlabeled: # <-- for safety
            label[:] = 0
        #label[label>self.config.num_cls-1] = 0
        # print("before",image.min(), image.max())
        # image = (image - image.min()) / (image.max() - image.min())
        if self.task == 'chd' or self.task == 'covid':
            min_val = np.percentile(image, 5)
            max_val = np.percentile(image, 95)
            image = image.clip(min=min_val, max=max_val)
        elif self.task == 'colon'  :
            image = image.clip(min=-250, max=275)
        elif self.task == 'synapse':
            image = image.clip(min=-75, max=275)
        elif self.task == 'word':
            image = image.clip(min=-200, max=250)
            
        image = (image - image.min()) / (image.max() - image.min())
        # image = (image - image.mean()) / (image.std() + 1e-8)
        # print("after",image.min(), image.max())
        # print("ss",image.max())
        # image = image.astype(np.float32)
        # label = label.astype(np.int8)

        # print(image.shape, label.shape)

        sample = {'image': image, 'label': label}
        '''
        if self.unlabeled == False:
            import torch
            torch.save(sample.copy(), 'sample.pth')
            return False
        #print(label.max())
        # print(sample['image'])
        '''
        if self.transform:
            # if not self.unlabeled and not self.is_val:
            #     sample = self.transform(sample, weights=self.transform.weights)
            # else:
            sample = self.transform(sample)

        return sample
