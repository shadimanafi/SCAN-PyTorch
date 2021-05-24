"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
test_percent=0.2
shuffle_dataset=True
from torch.utils.data.sampler import SubsetRandomSampler
datasetType=1
# datasetType=3

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

class CustomMixDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_folder = CustomImageFolder(root, transform)
        self.attr_tensor = self.get_tensor(root)

    def __getitem__(self, index):
        return [self.image_folder.__getitem__(index), self.attr_tensor[index], self.keys]

    def __len__(self):
        return self.len

    def get_tensor(self, root):
        if(datasetType==1):
            attr_file = open(os.path.join(root, 'Anno/list_attr_celeba.txt'), 'r')
            lines = attr_file.readlines()
            self.len = int(lines.pop(0))
            def isnt_punct(w):
                return not w in ['', ' ', ',', '\n']
            self.keys = list(map(lambda x: x.lstrip(' '), list(filter(isnt_punct, lines.pop(0).split(' ')))))
            self.n_key = len(self.keys)
            attr_tensor = []
            pbar = tqdm(total=self.len)
            pbar.set_description('[Loading Dataset]')
            for line in lines:
                pbar.update(1)
                words = [word for word in line.split(' ')[1:] if word!='' and word!='\n']
                vector = list(map(lambda x: (1 + float(x)) / 2, words))
                vector = np.array(vector)
                vector.resize([1, self.n_key])
                attr_tensor.append(vector)
            attr_tensor = np.concatenate(attr_tensor)
            pbar.write('[Dataset Loading Finished]')
            pbar.close()

            return attr_tensor

        elif(datasetType==3):
            attr_file = open(os.path.join(root, 'Anno/componentsLabels.txt'), 'r')
            lines = attr_file.readlines()
            self.len = len(lines)-1

            def isnt_punct(w):
                return not w in ['', ' ', ',', '\n']

            self.keys = list(map(lambda x: x.lstrip(';'), list(filter(isnt_punct, lines.pop(0).split(';')))))
            self.n_key = len(self.keys)
            attr_tensor = []
            pbar = tqdm(total=self.len)
            pbar.set_description('[Loading Dataset]')
            for line in lines:
                pbar.update(1)
                words = [word for word in line.split(';')[0:] if word != '' and word != '\n']
                # vector = list(map(lambda x: (1 + float(x)) / 2, words))
                vector=words
                vector = np.array(vector)
                vector.resize([1, self.n_key])
                attr_tensor.append(vector)
            attr_tensor = np.concatenate(attr_tensor)
            pbar.write('[Dataset Loading Finished]')
            pbar.close()

            return attr_tensor


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args, require_attr=False):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder if not require_attr else CustomMixDataset

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif name.lower() == 'furniture':
        root = os.path.join(dset_dir, 'BigFurniturePack')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder if not require_attr else CustomMixDataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

def return_data_test(args, require_attr=False):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder if not require_attr else CustomMixDataset

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif name.lower() == 'furniture':
        root = os.path.join(dset_dir, 'BigFurniturePack')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder if not require_attr else CustomMixDataset

    else:
        raise NotImplementedError


    data = dset(**train_kwargs)
    dataset_size=len(data)

    split= int(np.floor(test_percent * dataset_size))
    indices = list(range(dataset_size))
    if shuffle_dataset:
        random_seed = 42
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    test_loader = DataLoader(data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=test_sampler)

    data_loader = train_loader

    return train_loader,data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    if(datasetType==1):
        dset = CustomImageFolder('data/CelebA', transform)
    elif(datasetType==3):
        dset = CustomImageFolder('data/BigFurniturePack', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
