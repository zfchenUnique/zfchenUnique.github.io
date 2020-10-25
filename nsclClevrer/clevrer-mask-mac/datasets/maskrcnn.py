from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import pdb

class MaskRcnnDataset(Dataset):

    def __init__(self, object_features_path: str, video_features_path: str, type: str, split: str, video_split_info_path: str , use_attr_flag=False):
        self.object_features_path = object_features_path
        self.video_features_path = video_features_path

        with open(f'data/{type}/data.pkl', 'rb') as f:
            data = pickle.load(f)
        #split_info_full_path = os.path.join(video_split_info_path, split+'.txt')
        #self.split_vid_list = np.loadtxt(split_info_full_path).astype(np.int).tolist()
        ignore_list = []
        ignore_list += list(range(84, 89))
        ignore_list += list(range(345, 358))
        ignore_list += list(range(367, 371))
        self.valid_vid_list = [vid for vid  in range(1, 517) if vid not in ignore_list]
        self.use_attr_flag = use_attr_flag 

        self.split = split
        self.data = data[split]

        self.video_features = np.load(video_features_path, mmap_mode='r')
        if self.use_attr_flag:
            self.object_features = np.load(object_features_path, mmap_mode='r')

    def __getitem__(self, index):
        data = self.data[index]
        video_index = data['video_index']
        question_ids = data['question_ids']
        question_len = data['question_len']
        answer_id = data['answer_id']
        family_id = data['family_id']
        vid_ftr_index = self.valid_vid_list.index(video_index)
        video_features = self._get_video_features(vid_ftr_index)
        if self.use_attr_flag:
            object_features = self._get_object_features(vid_ftr_index)
            features = torch.cat((video_features, object_features), dim=0)
        else:
            features = video_features 
        question_ids = torch.LongTensor(question_ids)

        return features, question_ids, question_len, answer_id, family_id

    def _get_object_features(self, index: int):
        object_features = self.object_features[index]
        object_features = torch.from_numpy(object_features)
        return object_features

    def _get_video_features(self, index: int):
        video_features = self.video_features[index]
        video_features = torch.from_numpy(video_features)
        return video_features

    def __len__(self):
        return len(self.data)


def collate_data(batch: list):
    # Sort by question_length
    batch.sort(key=lambda x: x[2], reverse=True)

    features, question_ids, question_len, answer_id, family_id = zip(*batch)

    # [FloatTensor] -> FloatTensor
    features = torch.stack(features)
    # [[int], [int]] -> LongTensor
    question_ids = pad_sequence(question_ids, batch_first=True)

    question_len = torch.LongTensor(question_len)
    # [int] -> LongTensor
    answer_id = torch.LongTensor(answer_id)

    # [int] -> LongTensor
    family_id = torch.LongTensor(family_id)

    return features, question_ids, question_len, answer_id, family_id
