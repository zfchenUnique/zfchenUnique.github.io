import h5py
import argparse
import os
import pickle
from glob import glob
from typing import Dict, List
from pytorch_pretrained_bert import BertTokenizer


class Dictionary:

    def __init__(self):
        self.index2word = []
        self.word2index = {}
        self.index = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def add_sentence(self, sentence: str) -> List[int]:
        result = []
        for word in self.tokenizer.tokenize(sentence):
            index = self.add_word(word)
            result.append(index)

        return result

    def add_word(self, word: str) -> int:
        if word not in self.word2index:
            self.word2index[word] = self.index
            self.index2word.append(word)
            self.index += 1

        return self.word2index[word]

    def save(self, path: str, type: str):
        with open(os.path.join(path, f'{type}_dictionary.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'word2index': self.word2index,
                    'index2word': self.index2word
                },
                f
            )


def main(args):
    pickle_files = glob(os.path.join(args.pickle_dir, '*.pkl'))

    print('pickle files: ', pickle_files)

    indexes = get_indexes_from_h5(args.h5)

    for pickle_file in pickle_files:
        process_question(pickle_file, indexes, args.output_dir)


def get_indexes_from_h5(h5_path: str) -> Dict[str, int]:
    with h5py.File(h5_path, 'r') as h5:
        indexes = {bstr.decode(): i for i, bstr in enumerate(h5['index'])}
        # indexes = {bstr.decode(): i for i, bstr in enumerate(h5['indexes'])}

    return indexes


def process_question(pickle_path: str, indexes: list, output_dir: str):
    question_type = os.path.basename(pickle_path).split('.')[0]
    print(question_type)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    question_dictionary = Dictionary()
    answer_dictionary = Dictionary()

    # 兼容测试数据
    # import ipdb; ipdb.set_trace()
    if type(data) == list:
        data = {'test': data}

    for split, splited_data in data.items():

        result = []

        for item in splited_data:
            question = item['question']
            answer = item['answer']
            image_id = indexes[item['image_id']]

            answer_id = answer_dictionary.add_word(answer)

            question_ids = question_dictionary.add_sentence(question)
            question_len = len(question_ids)

            result.append({
                'question': question,
                'answer': answer,
                'answer_id': answer_id,
                'question_ids': question_ids,
                'question_len': question_len,
                'image_id': image_id
            })

        with open(os.path.join(output_dir, f'{split}_{question_type}.pkl'), 'wb') as f:
            pickle.dump(result, f)

    question_dictionary.save(output_dir, question_type + '_question')
    answer_dictionary.save(output_dir, question_type + '_answer')
    print(question_type, len(answer_dictionary.index2word), answer_dictionary.index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pickle_dir',
                        default='/data/vision/billf/scratch/chihan/gqa/processed/experiment_questions/')
    parser.add_argument('--h5', required=True)
    parser.add_argument('-o', '--output_dir')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
