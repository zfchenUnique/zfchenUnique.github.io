import argparse
import os
import pickle
from tqdm import tqdm
from typing import Dict, List
import json
from pytorch_pretrained_bert import BertTokenizer

"""
[
    questions: [question, answer, question_family]
    sence_index: int
    video_filename: sim_xxxxx.mp4
]
"""


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


def process_open_ended_questions(path: str, output: str):
    result = {
        'train': [],
        'val': [],
        'test': []
    }

    question_dictionary = Dictionary()
    answer_dictionary = Dictionary()
    family_dictionary = Dictionary()

    with open(path, 'r') as f:
        data = json.load(f)

        for video in tqdm(data):
            questions = video['questions']
            video_index = video['scene_index']

            if video_index < 10000:
                split = 'train'
            elif video_index > 10000 and video_index < 15000:
                split = 'val'
            elif video_index > 15000:
                split = 'test'

            for qa_pair in questions:
                question = qa_pair['question']
                answer = qa_pair['answer']
                family = qa_pair['question_family']

                question_ids = question_dictionary.add_sentence(question)
                answer_id = answer_dictionary.add_word(answer)
                family_id = family_dictionary.add_word(family)

                result[split].append({
                    'question': question,
                    'answer': answer,
                    'family': family,
                    'question_ids': question_ids,
                    'question_len': len(question_ids),
                    'answer_id': answer_id,
                    'family_id': family_id,
                    'video_index': video_index
                })

    save_all(output, result, question_dictionary, answer_dictionary, family_dictionary)


def save_all(output: str, result: dict, question_dictionary: Dictionary, answer_dictionary: Dictionary,
             family_dictionary: Dictionary):
    with open(os.path.join(output, f'data.pkl'), 'wb') as f:
        pickle.dump(result, f)

    question_dictionary.save(output, 'question')
    answer_dictionary.save(output, 'answer')
    family_dictionary.save(output, 'family')


def process_multiple_choice_questions(path: str, output: str):
    result = {
        'train': [],
        'val': [],
        'test': []
    }

    question_dictionary = Dictionary()
    answer_dictionary = Dictionary()
    family_dictionary = Dictionary()

    with open(path, 'r') as f:
        data = json.load(f)

        for video in tqdm(data):
            questions = video['questions']
            video_index = video['scene_index']

            if video_index < 10000:
                split = 'train'
            elif video_index > 10000 and video_index < 15000:
                split = 'val'
            elif video_index > 15000:
                split = 'test'

            for qa_pair in questions:
                question = qa_pair['question']

                family = qa_pair['question_type']

                correct_answers = qa_pair['correct']
                wrong_answers = qa_pair['wrong']

                for correct_pair in correct_answers:
                    # [answer, programe]
                    correct_choice = correct_pair[0]
                    answer = 'correct'

                    question_choice = question + correct_choice
                    question_ids = question_dictionary.add_sentence(question_choice)
                    answer_id = answer_dictionary.add_word(answer)
                    family_id = family_dictionary.add_word(family)

                    result[split].append({
                        'question': question,
                        'choice': correct_choice,
                        'answer': answer,
                        'family': family,
                        'question_ids': question_ids,
                        'question_len': len(question_ids),
                        'answer_id': answer_id,
                        'family_id': family_id,
                        'video_index': video_index
                    })

                for wrong_pair in wrong_answers:
                    # [answer, programe]
                    wrong_choice = wrong_pair[0]
                    answer = 'wrong'

                    question_choice = question + wrong_choice
                    question_ids = question_dictionary.add_sentence(question_choice)
                    answer_id = answer_dictionary.add_word(answer)
                    family_id = family_dictionary.add_word(family)

                    result[split].append({
                        'question': question,
                        'choice': wrong_choice,
                        'answer': answer,
                        'family': family,
                        'question_ids': question_ids,
                        'question_len': len(question_ids),
                        'answer_id': answer_id,
                        'family_id': family_id,
                        'video_index': video_index
                    })

    save_all(output, result, question_dictionary, answer_dictionary, family_dictionary)


def main(args):
    if args.type == 'open_ended':
        path = os.path.join(args.output, args.type)
        os.makedirs(path, exist_ok=True)
        process_open_ended_questions(args.input, path)
    elif args.type == 'multiple_choice':
        path = os.path.join(args.output, args.type)
        os.makedirs(path, exist_ok=True)
        process_multiple_choice_questions(args.input, path)
    else:
        raise NotImplemented()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', default='data')
    parser.add_argument('-t', '--type', choices=['open_ended', 'multiple_choice'])

    args = parser.parse_args()

    main(args)
