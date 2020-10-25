from torchpie.config import config
from torchpie.logging import logger
import pickle
from models import get_model
from torchpie.experiment import resume, experiment_path
import torch
from torch import nn
from datasets.maskrcnn import MaskRcnnDataset, collate_data
from torch.utils.data import DataLoader
from collections import Counter
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

            logger.info('Epoch: %d [%d/%d]',
                        epoch + 1, i, loader_len)

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
    logger.info('Avg Acc: %.5f', avg_acc)

    if writer:
        writer.add_scalar('epoch_metric/val_avg_acc', avg_acc, epoch)
        for k, v in family_total.items():
            key = family_dictionary['index2word'][k]
            writer.add_scalar(f'val/{key}_acc', family_correct[k] / v, epoch)

    return avg_acc


if __name__ == '__main__':
    best_acc = 0.0

    n_epoch = config.get_int('n_epoch')
    batch_size = config.get_int('batch_size')
    num_workers = config.get_int('num_workers')

    if not experiment_path:
        logger.warning('No experiment path, will not save checkpoint, validation log or tensorboard log')

    with open(config.get_string('pickle.question'), 'rb') as f:
        question_dictionary = pickle.load(f)

    with open(config.get_string('pickle.answer'), 'rb') as f:
        answer_dictionary = pickle.load(f)

    with open(config.get_string('pickle.family'), 'rb') as f:
        family_dictionary = pickle.load(f)

    n_words = len(question_dictionary['index2word'])
    n_answers = len(answer_dictionary['index2word'])

    # model_config = config.get_config('model')
    # model_config.put('question_input.n_words', n_words)

    # net = get_model(config.get_string('arch'), n_words, n_answers).to(device)
    net_running = get_model(config.get_string('arch'), n_words, n_answers).to(device)

    # net = nn.DataParallel(net)
    net_running = nn.DataParallel(net_running)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)

    start_epoch = 0
    if resume:
        logger.info(f'using resume: {resume}')
        cp = torch.load(resume)
        # net.load_state_dict(cp['state_dict'])
        net_running.load_state_dict(cp['state_dict'])
        # optimizer.load_state_dict(cp['optimizer'])
        best_acc = cp['best_acc']
        start_epoch = cp['epoch']

    # accumulate(net_running, net, 0)

    # train_set = GQADataset(config.get_string('h5'), config.get_string('pickle.train'))
    train_set = MaskRcnnDataset(
        config.get_string('object_features_path'),
        config.get_string('video_features_path'),
        type=config.get_string('type'),
        split='train'
    )
    # train_loader = DataLoader(
    #     train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_data, shuffle=True,
    #     pin_memory=True
    # )

    # val_set = GQADataset(config.get_string('h5'), config.get_string('pickle.test'))
    val_set = MaskRcnnDataset(
        config.get_string('object_features_path'),
        config.get_string('video_features_path'),
        type=config.get_string('type'),
        split='test'
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_data, pin_memory=True
    )

    # writer = SummaryWriter(log_dir=experiment_path) if experiment_path else None
    writer = None

    # if writer:
    #     image, question, q_len, _ = next(iter(train_loader))
    #     image = image.to(device)
    #     question = question.to(device)
    #     writer.add_graph(net, (image, question, q_len))

    # for epoch in range(start_epoch, n_epoch):
        # acc = train(epoch, train_loader)
    acc = valid(start_epoch, val_loader)

        # if experiment_path:
        #     is_best = acc > best_acc
        #     best_acc = max(acc, best_acc)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': net_running.state_dict(),
        #     'best_acc': best_acc,
        #     'optimizer': optimizer.state_dict()
        # }, is_best, experiment_path)

        # check_point_filename = 'checkpoint_{}.model'.format(str(epoch + 1).zfill(2))
        # with open(os.path.join(experiment_path, check_point_filename), 'wb') as f:
        #     torch.save(net_running.state_dict(), f)

    if writer:
        writer.close()
