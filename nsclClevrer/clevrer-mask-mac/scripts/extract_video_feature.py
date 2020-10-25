import os
import argparse

import torch
from torchvision import transforms
from torchvision.models import ResNet, resnet50

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import time
import numpy as np
from numpy.lib.format import open_memmap
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root', default='/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames')
parser.add_argument(
    '--save_to', default='/data/vision/billf/scratch/chuang/resnet50_new20k_14x14')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]
)

sampler = [i for i in range(0, 125, 5)]


class MyDataset(Dataset):

    def __init__(self, root, transform=transform, sampler=sampler):
        self.root = root
        self.transform = transform
        self.sampler = sampler

    def __len__(self):
        return 20000 * 25

    def __getitem__(self, index):
        folder = index // 25
        frame = self.sampler[index % 25]
        folder_name = "sim_{:05d}".format(folder)
        frame_name = "frame_{:05d}.png".format(frame)
        filename = os.path.join(self.root, folder_name, frame_name)
        image = Image.open(filename).convert('RGB')

        # print(filename)
        if self.transform:
            image = self.transform(image)
        return image, index


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    # x = self.layer4(x)

    return x


def my_resnet():
    resnet = resnet50(pretrained=True).to(device)
    resnet.forward = forward.__get__(resnet, ResNet)
    # resnet = nn.DataParallel(resnet)
    resnet.eval()
    return resnet


def test():
    dataset = MyDataset(args.root)

    # image2 = dataset[2]
    # print(image2.shape)
    # os.makedirs(args.save_to, exist_ok=True)

    loader = DataLoader(dataset, num_workers=4, batch_size=25)
    resnet = my_resnet()

    start = time.time()

    fp = open_memmap(args.save_to, mode='w+', dtype=np.float32, shape=(20000, 1024, 25, 14, 14))

    with torch.no_grad():
        for index, (sample, target) in enumerate(loader):
            # print(sample.shape)
            # print(target)

            start1 = time.time()

            sample = sample.to(device)
            feature = resnet(sample)
            print(feature.shape, index)

            cpu_feature = feature.cpu()
            np_feature = cpu_feature.numpy()
            # filename = "{:05d}.npy".format(index)
            # filename = os.path.join(args.save_to, filename)
            # np.save(filename, np_feature)

            # [T, C, H, W] -> [C, T, H, W]
            fp[index] = np_feature.transpose((1, 0, 2, 3))

            end = time.time()

            print("duration: {}, total: {}".format(end - start1, end - start))

            # break

    del fp


if __name__ == '__main__':
    test()
