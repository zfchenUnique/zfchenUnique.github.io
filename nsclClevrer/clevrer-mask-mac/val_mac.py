import os
import pickle
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchpie.config import config
from torchpie.experiment import experiment_path, resume
from torchpie.logging import logger

from datasets.gqa import GQADataset, collate_data
from models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def valid(epoch, dataloader):
    net_running.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer in dataloader:
            image, question = image.to(device), question.to(device)

            output = net_running(image, question, q_len)
            correct = output.argmax(1) == answer.to(device)

            family_correct[0] += correct.sum().item()
            family_total[0] += answer.shape[0]

            print(output.argmax(1))
            # for c, fam in zip(correct, family):
            #     fam = fam.item()
            #     if c:
            #         family_correct[fam] += 1
            #     family_total[fam] += 1
    if experiment_path:
        log_filename = 'log_{}.txt'.format(str(epoch + 1).zfill(2))
        with open(os.path.join(experiment_path, log_filename), 'w') as w:
            for k, v in family_total.items():
                w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    avg_acc = sum(family_correct.values()) / sum(family_total.values())
    logger.info('Avg Acc: %.5f', avg_acc)


if __name__ == '__main__':
    n_epoch = config.get_int('n_epoch')
    batch_size = config.get_int('batch_size')

    if not experiment_path:
        logger.warning('No experiment path, will not save checkpoint, validation log or tensorboard log')

    with open(config.get_string('pickle.question'), 'rb') as f:
        question_dictionary = pickle.load(f)

    with open(config.get_string('pickle.answer'), 'rb') as f:
        answer_dictionary = pickle.load(f)

    n_words = len(question_dictionary['index2word'])
    n_answers = len(answer_dictionary['index2word'])

    # model_config = config.get_config('model')
    # model_config.put('question_input.n_words', n_words)

    net_running = get_model(config.get_string('arch'), n_words, n_answers).to(device)

    net_running.load_state_dict(torch.load(resume))

    criterion = nn.CrossEntropyLoss()

    val_set = GQADataset(config.get_string('h5'), config.get_string('pickle.test'))
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=4, collate_fn=collate_data
    )

    # if writer:
    #     image, question, q_len, _ = next(iter(train_loader))
    #     image = image.to(device)
    #     question = question.to(device)
    #     writer.add_graph(net, (image, question, q_len))

    # for epoch in range(n_epoch):

    valid(0, val_loader)
