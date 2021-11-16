import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net import LinXiaoNet
from net.loss.alpha_loss import AlphaLoss
from net.data.chess_dataset import ChessDataset
from net.config import LinXiaoNetConfig
from utils.log import MyNetLogger

import os


def save_checkpoint(save_dir, ep_num, model_dict, optimizer_dict, lr_schedule_dict):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model_dict, os.path.join(save_dir, 'model.pth'))
    torch.save(optimizer_dict, os.path.join(save_dir, 'optimizer.pth'))
    torch.save(lr_schedule_dict, os.path.join(save_dir, 'lr_schedule.pth'))
    with open(os.path.join(save_dir, 'epoch.txt'), 'w+') as f:
        f.write('{}\n'.format(ep_num))


if __name__ == '__main__':

    conf = LinXiaoNetConfig()
    conf.set_cuda(True)
    conf.set_input_shape(16, 16)
    conf.set_train_info(10, 16, 1e-3)
    conf.set_checkpoint_config(1, 'checkpoints')
    conf.set_dataset_dir('out/m10000')
    conf.set_num_worker(0)
    conf.set_log('log/train.log')
    # conf.set_pretrained_path()

    logger = print if conf.log_file is None else MyNetLogger.default(conf.log_file)
    logger(conf)

    device = 'cuda' if conf.use_cuda else 'cpu'

    model = LinXiaoNet(3)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), conf.init_lr, 0.9, weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95)

    loss_func = AlphaLoss()
    loss_func.to(device)

    train_dataset = ChessDataset(conf.dataset_dir)
    train_loader = DataLoader(train_dataset, conf.batch_size, shuffle=False, num_workers=conf.num_worker)

    ep_num = 0

    if conf.pretrain_path is not None:
        filename_list = os.listdir(conf.pretrain_path)
        model_filename = None
        optimizer_filename = None
        lr_schedule_filename = None
        epoch_filename = None
        for filename in filename_list:
            if filename.find('model') > -1:
                model_filename = filename
            if filename.find('optimizer') > -1:
                optimizer_filename = filename
            if filename.find('lr_schedule') > -1:
                lr_schedule_filename = filename
            if filename.find('epoch') > -1:
                epoch_filename = filename
        if model_filename is not None:
            model.load_state_dict(torch.load(os.path.join(conf.pretrain_path, model_filename)))
        if optimizer_filename is not None:
            optimizer.load_state_dict(torch.load(os.path.join(conf.pretrain_path, optimizer_filename)))
        if lr_schedule_filename is not None:
            lr_schedule.load_state_dict(torch.load(os.path.join(conf.pretrain_path, lr_schedule_filename)))
        if epoch_filename is not None:
            with open(os.path.join(conf.pretrain_path, epoch_filename), 'r') as f:
                ep_num = int(f.readlines()[0].strip())
        logger('successfully load pretrained : {}'.format(conf.pretrain_path))

    for _ in range(ep_num, conf.epoch_num):
        model.train()
        for bat_state, bat_prob, bat_value in train_loader:
            bat_state, bat_prob, bat_value = bat_state.to(device), bat_prob.to(device), bat_value.to(device)
            optimizer.zero_grad()
            out_prob, out_value = model(bat_state)
            loss = loss_func(out_prob, out_value, bat_prob, bat_value)
            logger('train loss: {}'.format(loss.item()))
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        lr_schedule.step()
        ep_num += 1
        if conf.checkpoint_save_every_num is not None and conf.checkpoint_save_dir is not None and ep_num % conf.checkpoint_save_every_num == 0:
            save_checkpoint(
                os.path.join(conf.checkpoint_save_dir, 'epoch{}'.format(ep_num)),
                ep_num,
                model.state_dict(),
                optimizer.state_dict(),
                lr_schedule.state_dict()
            )
        # break


