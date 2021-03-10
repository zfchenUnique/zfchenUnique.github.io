import os
import json
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True)


def main(args):
    
    with open(args.filepath) as f:
        preds = json.load(f)

    with open('test_answers.json') as f:
        answers = json.load(f)

    stats = {
        'descriptive': [0, 0],
        'explanatory': [0, 0],
        'explanatory_choice': [0, 0],
        'predictive': [0, 0],
        'predictive_choice': [0, 0],
        'counterfactual': [0, 0],
        'counterfactual_choice': [0, 0],
    }

    #pdb.set_trace()

    #for i in range(5000):
    for i in range(len(preds)):
        pred = preds[i]
        ans = answers[i]
        if pred['scene_index'] != ans['scene_index']:
            raise ValueError('Scene index does not match.')
        
        q_dict = {pq['question_id']: pq for pq in pred['questions']}
        #for q in ans['questions']:
        pred_len = len(q_dict)
        #pdb.set_trace()
        for q in ans['questions'][:pred_len]:
            correct = True
            if q['question_id'] in q_dict:
                if q['question_type'] == 'descriptive':
                    correct = (q['answer'] == q_dict[q['question_id']]['answer'])
                else:
                    c_dict = {c['choice_id']: c for c in q_dict[q['question_id']]['choices']}
                    for c in q['choices']:
                        if c['choice_id'] in c_dict and c_dict[c['choice_id']]['answer'] == c['answer']:
                        #if c_dict[c['choice_id']]['answer'] == c['answer']:
                            stats['{}_choice'.format(q['question_type'])][0] += 1
                        else:
                            correct = False
                        stats['{}_choice'.format(q['question_type'])][1] += 1
            else:
                correct = False
            
            if correct:
                stats[q['question_type']][0] += 1
            stats[q['question_type']][1] += 1
    for k, v in stats.items():
        if k == 'descriptive':
            print('Descriptive: {:.2f}% ({:d} / {:d})'.format(
                float(v[0]) / v[1] * 100, v[0], v[1]))
        elif not k.endswith('_choice'):
            print('{}:'.format(k.capitalize()))
            print('    - Per choice: {:.2f}% ({:d} / {:d})'.format(
                float(stats[k+'_choice'][0]) / stats[k+'_choice'][1] * 100,
                stats[k+'_choice'][0], stats[k+'_choice'][1]))
            print('    - Per question: {:.2f}% ({:d} / {:d})'.format(
                float(v[0]) / v[1] * 100, v[0], v[1]))  


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

