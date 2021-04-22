"""
Run symbolic reasoning on open-ended questions
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
parser.add_argument('--use_in', default=0, type=int)  # Interaction network for dynamics prediction
args = parser.parse_args()


if args.use_event_ann != 0:
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision_old'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision_old'
if args.use_in:
    raw_motion_dir = 'data/propnet_preds/interaction_network'

question_path = 'data/questions/open_ended_questions.json'
program_path = 'data/parsed_programs/oe_{}pg.json'.format(args.n_progs)

#with open(program_path) as f:
#    parsed_pgs = json.load(f)
#with open(question_path) as f:
#    anns = json.load(f)


#train_ann_path = '../clevrer/train.json'
train_ann_path = '../clevrer/questions/train.json'
raw_motion_dir = '../temporal_reasoning-master/propnet_attr_noEdgeSuperv'
anns = jsonload(train_ann_path)
parsed_pgs = {}

total, correct = 0, 0
#pbar = tqdm(range(15000, 20000))
pbar = tqdm(range(0, 15000))
#for ann_idx in pbar:



for ann_idx in range(0, 100):
    question_scene = anns[ann_idx]
    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % ann_idx)

    sim = Simulation(ann_path, use_event_ann=(args.use_event_ann != 0))
    exe = Executor(sim)

    for q_idx, q in enumerate(question_scene['questions']):
        question = q['question']
        #parsed_pg = parsed_pgs[str(ann_idx)][str(q_idx)][0]
        parsed_pg = q['program']
        pred = exe.run(parsed_pg, debug=True)
        ans = q['answer']
        pdb.set_trace()
        if pred == ans:
            correct += 1
        total += 1

    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))

