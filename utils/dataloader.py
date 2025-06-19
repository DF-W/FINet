import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from scipy.ndimage import distance_transform_edt as distance


class MyDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, train_path, image_size, augmentations):
        self.image_size = image_size
        self.augmentations = augmentations

        image_root = os.path.join(train_path, 'images')
        gt_root = os.path.join(train_path, 'masks')
        egs_path = os.path.join(train_path, 'edges')

        self.images = []
        self.gts = []
        self.egs = []

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG') or f.endswith('.tif')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.PNG') or f.endswith('.tif')]
        self.egs = [os.path.join(egs_path, f) for f in os.listdir(egs_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.PNG')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.egs = sorted(self.egs)
        
        self.filter_files()
        self.size = len(self.images)

        if self.augmentations:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()])
            self.eg_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()])
            
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()])
            
            self.eg_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.gray_loader(self.gts[index])
        eg = self.binary_loader(self.egs[index])
        
        seed = np.random.randint(2023) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
            
            
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.eg_transform is not None:
            eg = self.eg_transform(eg)
            
        return image, gt, eg

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('1')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.image_size or w < self.image_size:
            h = max(h, self.image_size)
            w = max(w, self.image_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size
    
    def one_hot2dist(self, seg: np.ndarray) -> np.ndarray:
        # assert one_hot(torch.Tensor(seg), axis=0)
        # C: int = len(seg)

        res = np.zeros_like(seg)
        posmask = seg[0].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[0] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return res


def get_loader(train_path, batch_size, image_size, shuffle=True, num_workers=16, pin_memory=False, augmentation=False):

    dataset = MyDataset(train_path, image_size, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset(data.Dataset):
    def __init__(self, test_path, image_size):
        self.image_size = image_size

        image_root = os.path.join(test_path, 'images')
        gt_root = os.path.join(test_path, 'masks')

        self.images = []
        self.gts = []

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG') or f.endswith('.bmp')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.PNG') or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image) 
        gt = self.gray_loader(self.gts[index])
        gt = self.gt_transform(gt)
        name = self.images[index] 
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
