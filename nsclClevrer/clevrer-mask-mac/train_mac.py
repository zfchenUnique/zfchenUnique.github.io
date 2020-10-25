import os
import pickle
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchpie.config import config
from torchpie.experiment import experiment_path
from torchpie.logging import logger
from torchpie.checkpoint.saver import save_checkpoint

from datasets.gqa import GQADataset, collate_data
from models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


global_train_step = 0


def train(epoch, dataloader):
    global global_train_step
    train_step = 0
    moving_acc = None
    correct_count = 0
    total_count = 0
    total_loss = 0.

    net.train(True)
    for image, question, q_len, answer in dataloader:
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct = output.argmax(1) == answer
            step_correct_count = correct.sum().item()
            step_batch_size = correct.size(0)
            acc = step_correct_count / step_batch_size

        # if writer:
        #     writer.add_scalar('step_metric/loss', loss, global_train_step)
        #     writer.add_scalar('step_metric/acc', acc, global_train_step)

        if moving_acc is None:
            moving_acc = acc

        else:
            moving_acc = moving_acc * 0.99 + acc * 0.01

        logger.info('Epoch: %d; Loss: %.5f; Acc: %.5f',
                    epoch + 1, loss.item(), moving_acc)

        accumulate(net_running, net)
        global_train_step += 1
        train_step += 1
        correct_count += step_correct_count
        total_count += step_batch_size
        total_loss += loss.item()

    avg_loss = total_loss / train_step
    avg_acc = correct_count / total_count
    logger.info('Epoch %d finished. Average loss: %.5f Average acc: %.5f', epoch + 1, avg_loss, avg_acc)
    if writer:
        writer.add_scalar('epoch_metric/train_avg_loss', avg_loss, epoch)
        writer.add_scalar('epoch_metric/train_avg_acc', avg_acc, epoch)


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

    if writer:
        writer.add_scalar('epoch_metric/val_avg_acc', avg_acc, epoch)

    return avg_acc


if __name__ == '__main__':
    best_acc = 0.0

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

    net = get_model(config.get_string('arch'), n_words, n_answers).to(device)
    net_running = get_model(config.get_string('arch'), n_words, n_answers).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_set = GQADataset(config.get_string('h5'), config.get_string('pickle.train'))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=4, collate_fn=collate_data, shuffle=True
    )

    val_set = GQADataset(config.get_string('h5'), config.get_string('pickle.test'))
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=4, collate_fn=collate_data
    )

    writer = SummaryWriter(log_dir=experiment_path) if experiment_path else None

    # if writer:
    #     image, question, q_len, _ = next(iter(train_loader))
    #     image = image.to(device)
    #     question = question.to(device)
    #     writer.add_graph(net, (image, question, q_len))

    for epoch in range(n_epoch):
        train(epoch, train_loader)
        acc = valid(epoch, val_loader)

        if experiment_path:
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net_running.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best, experiment_path)

            # check_point_filename = 'checkpoint_{}.model'.format(str(epoch + 1).zfill(2))
            # with open(os.path.join(experiment_path, check_point_filename), 'wb') as f:
            #     torch.save(net_running.state_dict(), f)

    if writer:
        writer.close()
