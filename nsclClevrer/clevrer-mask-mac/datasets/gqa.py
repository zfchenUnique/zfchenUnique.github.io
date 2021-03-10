from torch.utils.data import Dataset
import h5py
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle
import numpy as np


class GQADataset(Dataset):

    def __init__(self, h5_path, pickle_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.features = self.h5['feature_map']

        with open(pickle_path, 'rb') as f:
            self.questions = pickle.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        image_id = question['image_id']
        question_ids = question['question_ids']
        question_len = question['question_len']
        answer = question['answer_id']

        image = self.features[image_id].astype(np.float32)

        return torch.from_numpy(image), torch.LongTensor(question_ids), question_len, answer


def collate_data(batch: list):
    # Sort by question_length
    batch.sort(key=lambda x: x[2], reverse=True)

    images, questions, question_lengths, answers = zip(*batch)

    # [FloatTensor] -> FloatTensor
    images = torch.stack(images).permute(0, 3, 1, 2)
    # [[int], [int]] -> LongTensor
    questions = pad_sequence(questions)

    question_lengths = torch.LongTensor(question_lengths)
    # [int] -> LongTensor
    answers = torch.LongTensor(answers)

    return images, questions, question_lengths, answers
