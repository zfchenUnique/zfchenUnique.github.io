import os
import pickle
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
#from torchpie.config import config
#from torchpie.experiment import experiment_path, resume
#from torchpie.logging import logger
#from torchpie.checkpoint.saver import save_checkpoint

# from datasets.gqa import GQADataset, collate_data
from datasets.maskrcnn import MaskRcnnDataset, collate_data
from models import get_model
import pdb

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

set_debugger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

experiment_path = config.get('DEFAULT', 'experiment_path')
resume = config.get('DEFAULT', 'resume')

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

    loader_len = len(dataloader)

    net.train(True)
    for i, (image, question, q_len, answer, _family) in enumerate(dataloader):
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        output = net(image, question, q_len)
        loss = criterion(output, answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct = output.argmax(1) == answer
            step_correct_count = correct.sum().item()
            step_batch_size = correct.size(0)
            acc = step_correct_count / step_batch_size

        if moving_acc is None:
            moving_acc = acc

        else:
            moving_acc = moving_acc * 0.99 + acc * 0.01

        if i %config.getint('DEFAULT', 'log_every_iter') ==0:
            print('Epoch: %d [%d/%d]; Loss: %.5f; Acc: %.5f'
                    %(epoch + 1, i, loader_len, loss.item(), moving_acc))

        accumulate(net_running, net)
        global_train_step += 1
        train_step += 1
        correct_count += step_correct_count
        total_count += step_batch_size
        total_loss += loss.item()

    avg_loss = total_loss / train_step
    avg_acc = correct_count / total_count
    print('Epoch %d finished. Average loss: %.5f Average acc: %.5f' 
            % (epoch + 1, avg_loss, avg_acc))
    if writer:
        writer.add_scalar('epoch_metric/train_avg_loss', avg_loss, epoch)
        writer.add_scalar('epoch_metric/train_avg_acc', avg_acc, epoch)

    return avg_acc


def valid(epoch, dataloader):
    net_running.train(False)
    family_correct = Counter()
    family_total = Counter()

    loader_len = len(dataloader)
    result = []

    with torch.no_grad():

        for i, (image, question, q_len, answer, family) in enumerate(dataloader):
            image, question = image.to(device), question.to(device)

            output = net_running(image, question, q_len)
            correct = output.argmax(1) == answer.to(device)

            # family_correct[0] += correct.sum().item()
            # family_total[0] += answer.shape[0]
            for c, fam in zip(correct, family):
                fam = fam.item()
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

            print('Epoch: %d [%d/%d]' 
                    % (epoch + 1, i, loader_len))

            result.append(correct.cpu())

    if experiment_path:
        log_filename = 'log_{}.txt'.format(str(epoch + 1).zfill(2))
        with open(os.path.join(experiment_path, log_filename), 'w') as w:
            for k, v in family_total.items():
                key = family_dictionary['index2word'][k]
                w.write('{}: {:.5f}\n'.format(key, family_correct[k] / v))

        result = torch.cat(result)
        torch.save(result, os.path.join(experiment_path, f'result_{epoch}.pth'))

    avg_acc = sum(family_correct.values()) / sum(family_total.values())
    print('Avg Acc: %.5f' %(avg_acc))

    if writer:
        writer.add_scalar('epoch_metric/val_avg_acc', avg_acc, epoch)
        for k, v in family_total.items():
            key = family_dictionary['index2word'][k]
            writer.add_scalar(f'val/{key}_acc', family_correct[k] / v, epoch)

    return avg_acc


if __name__ == '__main__':
    best_acc = 0.0
    n_epoch = config.getint('DEFAULT', 'n_epoch')
    batch_size = config.getint('DEFAULT', 'batch_size')
    num_workers = config.getint('DEFAULT', 'num_workers')

    if not experiment_path:
        print('No experiment path, will not save checkpoint, validation log or tensorboard log')

    with open(config.get('pickle', 'question'), 'rb') as f:
        question_dictionary = pickle.load(f)

    with open(config.get('pickle', 'answer'), 'rb') as f:
        answer_dictionary = pickle.load(f)

    with open(config.get('pickle', 'family'), 'rb') as f:
        family_dictionary = pickle.load(f)

    n_words = len(question_dictionary['index2word'])
    n_answers = len(answer_dictionary['index2word'])

    # model_config = config.get_config('model')
    # model_config.put('question_input.n_words', n_words)

    net = get_model(config.get('model', 'arch'), n_words, n_answers).to(device)
    net_running = get_model(config.get('model', 'arch'), n_words, n_answers).to(device)

    net = nn.DataParallel(net)
    net_running = nn.DataParallel(net_running)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    start_epoch = 0
    if resume:
        print(f'using resume: {resume}')
        cp = torch.load(resume)
        net.load_state_dict(cp['state_dict'])
        net_running.load_state_dict(cp['state_dict'])
        # optimizer.load_state_dict(cp['optimizer'])
        best_acc = cp['best_acc']
        start_epoch = cp['epoch']

    accumulate(net_running, net, 0)

    # train_set = GQADataset(config.get_string('h5'), config.get_string('pickle.train'))
    train_set = MaskRcnnDataset(
    config.get('DEFAULT', 'object_features_path'),
        config.get('DEFAULT', 'video_features_path'),
        type=config.get('DEFAULT', 'type'),
        split='train',
        video_split_info_path=config.get('DEFAULT', 'video_split_info_path'),
        use_attr_flag=config.getboolean('DEFAULT', 'use_attr_flag')
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_data, shuffle=True,
        pin_memory=True
    )

    val_set = MaskRcnnDataset(
    config.get('DEFAULT', 'object_features_path'),
        config.get('DEFAULT', 'video_features_path'),
        type=config.get('DEFAULT', 'type'),
        split='val',
        video_split_info_path=config.get('DEFAULT', 'video_split_info_path'),
        use_attr_flag=config.getboolean('DEFAULT', 'use_attr_flag')
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_data, pin_memory=True
    )

    writer = SummaryWriter(log_dir=experiment_path) if experiment_path else None

    # if writer:
    #     image, question, q_len, _ = next(iter(train_loader))
    #     image = image.to(device)
    #     question = question.to(device)
    #     writer.add_graph(net, (image, question, q_len))

    for epoch in range(start_epoch, n_epoch):
        acc = train(epoch, train_loader)
        # acc = valid(epoch, val_loader)
        pdb.set_trace()
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
