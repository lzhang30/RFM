import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import read_list, read_nifti
from utils.config import Config



class Config:
    def __init__(self,task):
        self.task = task
        if task == 'ACC':
            self.base_dir = '' #path to the dataset
            self.save_dir = '' #path to save the data
            self.patch_size = (96, 128, 128)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50

config = Config('ACC')

print(config)


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    if not os.path.exists(os.path.join(config.save_dir, 'npy')):
        os.makedirs(os.path.join(config.save_dir, 'npy'))
    for tag in ['Tr', 'Va']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(config.base_dir, f'images{tag}', '*.nii.gz'))):
            print(path)
            img_id = path.split('/')[-1].split('.')[0]
            print(img_id)
            img_ids.append(img_id)
            # label_id = 'label'+ img_id[3:]

            image_path = os.path.join(config.base_dir, f'images{tag}', f'{img_id}.nii.gz')
            if config.task == 'colon':
                label_path =os.path.join(config.base_dir, f'labels{tag}', f'{img_id}.seg.nii.gz')
            else:  
                label_path =os.path.join(config.base_dir, f'labels{tag}', f'{img_id}.nii.gz')


            resize_shape=(config.patch_size[0]+config.patch_size[0]//4,
                          config.patch_size[1]+config.patch_size[1]//4,
                          config.patch_size[2]+config.patch_size[2]//4)

            image = read_nifti(image_path)
            print(label_path)
            try:
                label = read_nifti(label_path)
                
                islabel = True
            except:
                islabel = False
            image = image.astype(np.float32)
            if islabel:
                label = label.astype(np.int8)
            

            image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            if islabel:
                label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)

            image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
            if islabel:
                label = F.interpolate(label, size=resize_shape,mode='nearest')
            image = image.squeeze().numpy()
            if islabel:
                label = label.squeeze().numpy()


            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}_image.npy'),
                image
            )
            if islabel:
                np.save(
                    os.path.join(config.save_dir, 'npy', f'{img_id}_label.npy'),
                    label
                )





def process_split_fully(train_ratio=0.9):
    if not os.path.exists(os.path.join(config.save_dir, 'splits')):
        os.makedirs(os.path.join(config.save_dir, 'splits'))
    for tag in ['Tr', 'Va']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(config.base_dir, f'images{tag}', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('.')[0]
            img_ids.append(img_id)
        print(img_ids)
        
        if tag == 'Tr':
            train_val_ids = np.random.permutation(img_ids)
            # split_idx = int(len(img_ids) * train_ratio)
            # train_val_ids = img_ids[:split_idx]
            # test_ids = sorted(img_ids[split_idx:])

            # train_val_ids = [i for i in img_ids if i not in test_ids]
            split_idx = int(len(train_val_ids) * train_ratio)
            train_ids = sorted(train_val_ids[:split_idx])
            eval_ids = sorted(train_val_ids[split_idx:])
            write_txt(
                train_ids,
                os.path.join(config.save_dir, 'splits/train.txt')
            )
            write_txt(
                eval_ids,
                os.path.join(config.save_dir, 'splits/eval.txt')
            )

        else:
            test_ids = np.random.permutation(img_ids)
            test_ids = sorted(test_ids)
            write_txt(
                test_ids,
                os.path.join(config.save_dir, 'splits/test.txt')
            )


def process_split_semi(split='train', labeled_ratio=10):
    ids_list = read_list(split, task=config.task)
    ids_list = np.random.permutation(ids_list)

    split_idx = int(len(ids_list) * labeled_ratio/100)
    labeled_ids = sorted(ids_list[:split_idx])
    unlabeled_ids = sorted(ids_list[split_idx:])
    
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'splits/labeled_{labeled_ratio}p.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(config.save_dir, f'splits/unlabeled_{labeled_ratio}p.txt')
    )


if __name__ == '__main__':
    process_npy()
    process_split_fully()
    process_split_semi(labeled_ratio=2)
    process_split_semi(labeled_ratio=5)
    process_split_semi(labeled_ratio=20)
    process_split_semi(labeled_ratio=40)
    process_split_semi(labeled_ratio=50)
