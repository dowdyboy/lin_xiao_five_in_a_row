import os

import torch
from torch.utils.data import DataLoader

from mcts.config import MCTSConfig
from mcts import mcts_gen_chess
from net import LinXiaoNet
from net.config import LinXiaoNetConfig
from net.data.chess_dataset import ChessDataset
from net.loss.alpha_loss import AlphaLoss
from utils.log import MyNetLogger


def save_checkpoint(save_dir, chess_num, ep_num, model_dict, optimizer_dict, lr_schedule_dict):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model_dict, os.path.join(save_dir, 'model.pth'))
    torch.save(optimizer_dict, os.path.join(save_dir, 'optimizer.pth'))
    torch.save(lr_schedule_dict, os.path.join(save_dir, 'lr_schedule.pth'))
    with open(os.path.join(save_dir, 'chess_num.txt'), 'w+') as f:
        f.write('{}\n'.format(chess_num))
    with open(os.path.join(save_dir, 'ep_num.txt'), 'w+') as f:
        f.write('{}\n'.format(ep_num))


if __name__ == '__main__':
    mcts_conf = MCTSConfig()
    mcts_conf.set_chess_config(10, 2000)
    mcts_conf.set_save_dir('out/step_m10000')
    mcts_conf.set_log_file('log/step_m10000.log')

    logger = print if mcts_conf.log_file is None else MyNetLogger.default(mcts_conf.log_file)
    logger(mcts_conf)

    net_conf = LinXiaoNetConfig()
    net_conf.set_cuda(True)
    net_conf.set_input_shape(16, 16)
    net_conf.set_train_info(90, 4, 1e-2)
    net_conf.set_checkpoint_config(90, 'checkpoints/step_m10000')
    net_conf.set_dataset_dir('out/step_m10000')
    net_conf.set_num_worker(0)
    # conf.set_log('log/train.log')
    net_conf.set_pretrained_path(None)
    logger(net_conf)

    device = 'cuda' if net_conf.use_cuda else 'cpu'
    device_cpu = 'cpu'

    model = LinXiaoNet(3)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), net_conf.init_lr, 0.9, weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.95)

    loss_func = AlphaLoss()
    loss_func.to(device)

    chess_num = 0
    ep_num = 0

    if net_conf.pretrain_path is not None:
        filename_list = os.listdir(net_conf.pretrain_path)
        model_filename = None
        optimizer_filename = None
        lr_schedule_filename = None
        chess_num_filename = None
        ep_num_filename = None
        for filename in filename_list:
            if filename.find('model') > -1:
                model_filename = filename
            if filename.find('optimizer') > -1:
                optimizer_filename = filename
            if filename.find('lr_schedule') > -1:
                lr_schedule_filename = filename
            if filename.find('chess_num') > -1:
                chess_num_filename = filename
            if filename.find('ep_num') > -1:
                ep_num_filename = filename
        if model_filename is not None:
            model.load_state_dict(torch.load(os.path.join(net_conf.pretrain_path, model_filename)))
        if optimizer_filename is not None:
            optimizer.load_state_dict(torch.load(os.path.join(net_conf.pretrain_path, optimizer_filename)))
        if lr_schedule_filename is not None:
            lr_schedule.load_state_dict(torch.load(os.path.join(net_conf.pretrain_path, lr_schedule_filename)))
        if chess_num_filename is not None:
            with open(os.path.join(net_conf.pretrain_path, chess_num_filename), 'r') as f:
                chess_num = int(f.readlines()[0].strip())
        if ep_num_filename is not None:
            with open(os.path.join(net_conf.pretrain_path, ep_num_filename), 'r') as f:
                ep_num = int(f.readlines()[0].strip())
        logger('successfully load pretrained : {}'.format(net_conf.pretrain_path))

    for _ in range(chess_num, mcts_conf.num_chess):
        model.to(device_cpu)
        mcts_gen_chess(model, chess_num + 1, mcts_conf.simulate_count, mcts_conf.save_dir, mcts_conf.log_file, logger)
        model.to(device)
        chess_num += 1
        train_dataset = ChessDataset(net_conf.dataset_dir, 1, logger)
        train_loader = DataLoader(train_dataset, net_conf.batch_size, shuffle=True, num_workers=net_conf.num_worker)
        for _ in range(ep_num, ep_num + net_conf.epoch_num):
            logger('epoch {} start...'.format(ep_num))
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
            if net_conf.checkpoint_save_every_num is not None and net_conf.checkpoint_save_dir is not None and ep_num % net_conf.checkpoint_save_every_num == 0:
                save_checkpoint(
                    os.path.join(net_conf.checkpoint_save_dir, 'epoch{}'.format(ep_num)),
                    chess_num,
                    ep_num,
                    model.state_dict(),
                    optimizer.state_dict(),
                    lr_schedule.state_dict()
                )



    pass
