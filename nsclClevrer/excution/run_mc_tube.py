"""
Run symbolic reasoning on multiple-choice questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation

import pdb
from utils import * 

set_debugger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_progs', required=True)
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Use interaction network
args = parser.parse_args()


if args.use_event_ann != 0:
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision_old'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision_old'
if args.use_in != 0:
    raw_motion_dir = 'data/propnet_preds/interaction_network'

question_path = 'data/questions/multiple_choice_questions.json'
if args.n_progs == 'all':
    program_path = 'data/parsed_programs/mc_allq_allc.json'
else:
    program_path = 'data/parsed_programs/mc_{}q_{}c.json'.format(args.n_progs, int(args.n_progs)*4)
# program_path = 'data/parsed_programs/mc_1000q_4000c.json'

print(raw_motion_dir)
print(program_path)

#with open(program_path) as f:
    #parsed_pgs = json.load(f)
#    parsed_pgs = {}
#with open(question_path) as f:
    #anns = json.load(f)
#    anns = {}

#val_ann_path = '../clevrer/validation.json'
val_ann_path = '../clevrer/train.json'
raw_motion_dir = '../temporal_reasoning-master/propnet_attr_noEdgeSuperv'
anns = jsonload(val_ann_path)


parsed_pgs = {}
#anns = {}

total, correct = 0, 0
total_per_q, correct_per_q = 0, 0
total_expl, correct_expl = 0, 0
total_expl_per_q, correct_expl_per_q = 0, 0
total_pred, correct_pred = 0, 0
total_pred_per_q, correct_pred_per_q = 0, 0
total_coun, correct_coun = 0, 0
total_coun_per_q, correct_coun_per_q = 0, 0
#pbar = tqdm(range(15000, 20000))
pbar = tqdm(range(0, 10000))
for ann_idx in pbar:
    question_scene = anns[ann_idx]


    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % ann_idx)

    sim = Simulation(ann_path, use_event_ann=(args.use_event_ann != 0))
    exe = Executor(sim)

    for q_idx, q in enumerate(question_scene['questions']):

        if 'choices' not in q.keys():
            continue 

        pdb.set_trace()
        question = q['question']
        q_ann = q['program']
        correct_question = True
        #for c in q_ann['choices']:
        for c in q['choices']:
            full_pg = c['program'] + q_ann
            ans = c['answer']
            pred = exe.run(full_pg, debug=True)
            if ans == pred:
                correct += 1
            else:
                correct_question = False
            total += 1

            if q['question_type'].startswith('explanatory'):
                if ans == pred:
                    correct_expl += 1
                total_expl += 1

            if q['question_type'].startswith('predictive'):
                # print(pred, ans)
                if ans == pred:
                    correct_pred += 1
                total_pred += 1

            if q['question_type'].startswith('counterfactual'):
                if ans == pred:
                    correct_coun += 1
                total_coun += 1

        if correct_question:
            correct_per_q += 1
        total_per_q += 1

        if q['question_type'].startswith('explanatory'):
            if correct_question:
                correct_expl_per_q += 1
            total_expl_per_q += 1

        if q['question_type'].startswith('predictive'):
            if correct_question:
                correct_pred_per_q += 1
            total_pred_per_q += 1

        if q['question_type'].startswith('counterfactual'):
            if correct_question:
                correct_coun_per_q += 1
            total_coun_per_q += 1

    # print('up to scene %d: %d / %d correct options, accuracy %f %%'
    #       % (ann_idx, correct, total, (float(correct)*100/total)))
    # print('up to scene %d: %d / %d correct questions, accuracy %f %%'
    #       % (ann_idx, correct_per_q, total_per_q, (float(correct_per_q)*100/total_per_q)))
    # print()
    pbar.set_description('per choice {:f}, per questions {:f}'.format(float(correct)*100/total, float(correct_per_q)*100/total_per_q))

print('============ results ============')
print('overall accuracy per option: %f %%' % (float(correct) * 100.0 / total))
print('overall accuracy per question: %f %%' % (float(correct_per_q) * 100.0 / total_per_q))
print('explanatory accuracy per option: %f %%' % (float(correct_expl) * 100.0 / total_expl))
print('explanatory accuracy per question: %f %%' % (float(correct_expl_per_q) * 100.0 / total_expl_per_q))
print('predictive accuracy per option: %f %%' % (float(correct_pred) * 100.0 / total_pred))
print('predictive accuracy per question: %f %%' % (float(correct_pred_per_q) * 100.0 / total_pred_per_q))
print('counterfactual accuracy per option: %f %%' % (float(correct_coun) * 100.0 / total_coun))
print('counterfactual accuracy per question: %f %%' % (float(correct_coun_per_q) * 100.0 / total_coun_per_q))
print('============ results ============')

output_ann = {
    'total_options': total,
    'correct_options': correct,
    'total_questions': total_per_q,
    'correct_questions': correct_per_q,
    'total_explanatory_options': total_expl,
    'correct_explanatory_options': correct_expl,
    'total_explanatory_questions': total_expl_per_q,
    'correct_explanatory_questions': correct_expl_per_q,
    'total_predictive_options': total_pred,
    'correct_predictive_options': correct_pred,
    'total_predictive_questions': total_pred_per_q,
    'correct_predictive_questions': correct_pred_per_q,
    'total_counterfactual_options': total_coun,
    'correct_counterfactual_options': correct_coun,
    'total_counterfactual_questions': total_coun_per_q,
    'correct_counterfactual_questions': correct_coun_per_q,
}

output_file = 'result_mc.json'
if args.use_in != 0:
    output_file = 'result_mc_in.json'
with open(output_file, 'w') as fout:
    json.dump(output_ann, fout)
