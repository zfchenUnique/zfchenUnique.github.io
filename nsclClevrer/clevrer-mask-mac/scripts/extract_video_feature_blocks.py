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
import pdb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root', default='/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames')
parser.add_argument(
    '--save_to', default='/data/vision/billf/scratch/chuang/resnet50_new20k_14x14')
parser.add_argument(
    '--sample_num', default=16, type=int)
parser.add_argument(
    '--video_num', default=494, type=int)
parser.add_argument(
    '--batch_size', default=16, type=int)
parser.add_argument(
    '--num_workers', default=4, type=int)
parser.add_argument(
    '--feat_dim', default=1024, type=int)
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

    def __init__(self, root, transform=transform, args=None):
        self.root = root
        self.transform = transform
        self.args = args
        ignore_list = []
        ignore_list += list(range(84, 89))
        ignore_list += list(range(345, 358))
        ignore_list += list(range(367, 371))
        self.valid_vid_list = [vid for vid  in range(1, 517) if vid not in ignore_list]
        self.W = 224
        self.H = 224

    def __len__(self):
        return len(self.valid_vid_list) 

    def __getitem__(self, index):
        vid = self.valid_vid_list[index]
        filename = os.path.join(self.root, 'img_concat_'+str(vid)+'.png')
        img_concat = Image.open(filename).convert('RGB')
        W, H_concat = img_concat.size
        frm_num = H_concat // self.H

        smp_diff = int(frm_num/self.args.sample_num)
        frm_offset =  int(smp_diff//2)
        frm_list = list(range(frm_offset, frm_num, smp_diff))
        img_list = [] 
        frm_id_list = []
        #pdb.set_trace()
        for i, frm in enumerate(frm_list):
            frm_id = frm 
            top = frm*self.H
            bottom = (frm+1)*self.H
            left  = 0 
            right = self.W
            img = img_concat.crop((left, top, right, bottom))
            W, H = img.size

            if self.transform:
                img = self.transform(img)
            img_list.append(img)
            frm_id_list.append(frm)
            if len(frm_id_list)>=self.args.sample_num:
                break 
        image = torch.stack(img_list, dim=0)
        return image, vid, frm_id_list

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

def collate_dict(batch):
    return batch

def test():
    dataset = MyDataset(args.root, args=args)
    loader = DataLoader(dataset, num_workers=args.num_workers, 
            batch_size=args.batch_size, collate_fn=collate_dict, shuffle=False)
    resnet = my_resnet()

    start = time.time()

    dir_folder = os.path.dirname(args.save_to)
    if not os.path.isdir(dir_folder):
        os.makedirs(dir_folder)
    assert args.video_num == len(dataset)
    fp = open_memmap(args.save_to, mode='w+', dtype=np.float32, 
            shape=(args.video_num, args.feat_dim, args.sample_num, 14, 14))
    save_to_info_path =  args.save_to.replace('.npy', '.pkl')
    info_list = []
    with torch.no_grad():
        for index, data_list in enumerate(loader):
            sample, vid, frm_id_list = data_list[0]
            info_list.append({'video_index': vid, 'frame_list':frm_id_list})
            start1 = time.time()

            sample = sample.to(device)
            feature = resnet(sample)
            print(feature.shape, index, vid)

            cpu_feature = feature.cpu()
            np_feature = cpu_feature.numpy()
            # filename = "{:05d}.npy".format(index)
            # filename = os.path.join(args.save_to, filename)
            # np.save(filename, np_feature)

            # [T, C, H, W] -> [C, T, H, W]
            fp[index] = np_feature.transpose((1, 0, 2, 3))

            end = time.time()

            print("duration: {}, total: {}".format(end - start1, end - start))
            #break

    del fp
    fh = open(save_to_info_path, 'wb')
    pickle.dump(info_list, fh) 

if __name__ == '__main__':
    test()
