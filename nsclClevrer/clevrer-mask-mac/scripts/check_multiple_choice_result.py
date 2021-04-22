import argparse
import torch
import json
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',
                        default='/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/questions/multiple_choice_questions.json')
    parser.add_argument('-r', '--result')

    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)

    data = data[15000: 20000]  # test set

    result = torch.load(args.result)

    family_correct = Counter()
    family_total = Counter()

    # 通过index: index+total_questions范围内是否全1来判断是否正确
    index = 0
    for video in data:
        questions = video['questions']

        for question in questions:
            num_correct = len(question['correct'])
            num_wrong = len(question['wrong'])
            family = question['question_type']
            total_questions = num_correct + num_wrong

            if result[index: index + total_questions].all():
                family_correct[family] += 1
            family_total[family] += 1

            # 更新index
            index += total_questions

    num_questions = 0
    correct_questions = 0
    for k, v in family_total.items():
        print('{}: {:.5f}\n'.format(k, family_correct[k] / v))
        correct_questions += family_correct[k]
        num_questions += v

    print('per quesiton: ', correct_questions / num_questions)

    print('total answers', result.shape[0])

    avg_acc = sum(family_correct.values()) / sum(family_total.values())

    print('per answer:', result.sum().item() / result.shape[0])
