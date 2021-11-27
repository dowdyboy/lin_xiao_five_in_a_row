import os
import pickle

import numpy as np
import torch

from net.config import LinXiaoNetConfig
from net import LinXiaoNet
from net.loss.alpha_loss import AlphaLoss
from net.data.train_data_cache import TrainDataCache
from mcts.monte_tree_v2 import MonteTree, transfer_to_net_input, pos_idx2pos_pair
from utils.log import init_logger, logger


def save_chess_record(file_path, record):
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb+') as f:
        pickle.dump(record, f)


def save_checkpoint(save_dir, ep_num, chess_num, model_dict, optimizer_dict, lr_schedule_dict, data_cache):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model_dict, os.path.join(save_dir, 'model.pth'))
    torch.save(optimizer_dict, os.path.join(save_dir, 'optimizer.pth'))
    torch.save(lr_schedule_dict, os.path.join(save_dir, 'lr_schedule.pth'))
    with open(os.path.join(save_dir, 'epoch.txt'), 'w+') as f:
        f.write('{}\n'.format(ep_num))
    with open(os.path.join(save_dir, 'chess_num.txt'), 'w+') as f:
        f.write('{}\n'.format(chess_num))
    with open(os.path.join(save_dir, 'data_cache.pkl'), 'wb+') as f:
        pickle.dump(data_cache, f)
    # with open(os.path.join(save_dir, 'tree.pkl'), 'wb+') as f:
    #     pickle.dump(tree, f)


def load_checkpoint(checkpoint_path):
    filename_list = os.listdir(checkpoint_path)
    model_filename = None
    optimizer_filename = None
    lr_schedule_filename = None
    epoch_filename = None
    chess_num_filename = None
    data_cache_filename = None
    # tree_filename = None

    model_data = None
    optimizer_data = None
    lr_schedule_data = None
    epoch_data = None
    chess_num = None
    data_cache_data = None
    # tree_data = None

    for filename in filename_list:
        if filename.find('model') > -1:
            model_filename = filename
        if filename.find('optimizer') > -1:
            optimizer_filename = filename
        if filename.find('lr_schedule') > -1:
            lr_schedule_filename = filename
        if filename.find('epoch') > -1:
            epoch_filename = filename
        if filename.find('chess_num') > -1:
            chess_num_filename = filename
        if filename.find('data_cache') > -1:
            data_cache_filename = filename
        # if filename.find('tree') > -1:
        #     tree_filename = filename
    if model_filename is not None:
        model_data = torch.load(os.path.join(checkpoint_path, model_filename))
    if optimizer_filename is not None:
        optimizer_data = torch.load(os.path.join(checkpoint_path, optimizer_filename))
    if lr_schedule_filename is not None:
        lr_schedule_data = torch.load(os.path.join(checkpoint_path, lr_schedule_filename))
    if epoch_filename is not None:
        with open(os.path.join(checkpoint_path, epoch_filename), 'r') as f:
            epoch_data = int(f.readlines()[0].strip())
    if chess_num_filename is not None:
        with open(os.path.join(checkpoint_path, chess_num_filename), 'r') as f:
            chess_num = int(f.readlines()[0].strip())
    if data_cache_filename is not None:
        with open(os.path.join(checkpoint_path, data_cache_filename), 'rb') as f:
            data_cache_data = pickle.load(f)
    # if tree_filename is not None:
    #     with open(os.path.join(checkpoint_path, tree_filename), 'rb') as f:
    #         tree_data = pickle.load(f)
    return model_data, optimizer_data, lr_schedule_data, data_cache_data, epoch_data, chess_num


def generate_train_data(chess_size, chess_record):
    chess_state = np.zeros((chess_size, chess_size))
    data = []
    player = 1
    winner = -1 if len(chess_record) % 2 == 0 else 1
    for i in range(len(chess_record)):
        pos_idx = chess_record[i][1]
        state = transfer_to_net_input(chess_state, player, chess_size)
        data.append({
            'state': state,
            'distribution': chess_record[i][0],
            'value': winner
        })
        chess_state[pos_idx2pos_pair(pos_idx, chess_size)[0], pos_idx2pos_pair(pos_idx, chess_size)[1]] = player
        player = -player
        # TODO: 思考这里为什么要变号
        winner = -winner
    return data


if __name__ == '__main__':

    conf = LinXiaoNetConfig()
    conf.set_cuda(True)
    conf.set_input_shape(16, 16)
    conf.set_train_info(5, 16, 1e-2)
    conf.set_checkpoint_config(5, 'checkpoints/v2m4000')
    conf.set_num_worker(0)
    conf.set_log('log/v2train.log')

    init_logger(conf.log_file)
    logger()(conf)

    device = 'cuda' if conf.use_cuda else 'cpu'

    model = LinXiaoNet(3)
    model.to(device)

    loss_func = AlphaLoss()
    loss_func.to(device)

    optimizer = torch.optim.SGD(model.parameters(), conf.init_lr, 0.9, weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95)

    tree = MonteTree(model, device, chess_size=conf.input_shape[0], simulate_count=100)
    data_cache = TrainDataCache(num_worker=conf.num_worker)

    ep_num = 0
    chess_num = 0
    train_every_chess = 2

    if conf.pretrain_path is not None:
        model_data, optimizer_data, lr_schedule_data, data_cache, ep_num, chess_num = load_checkpoint(conf.pretrain_path)
        model.load_state_dict(model_data)
        optimizer.load_state_dict(optimizer_data)
        lr_schedule.load_state_dict(lr_schedule_data)
        logger()('successfully load pretrained : {}'.format(conf.pretrain_path))

    while True:
        logger()(f'self chess game no.{chess_num+1} start.')
        chess_record = tree.self_game()
        logger()(f'self chess game no.{chess_num+1} end.')
        train_data = generate_train_data(tree.chess_size, chess_record)
        for i in range(len(train_data)):
            data_cache.push(train_data[i])
        if chess_num % train_every_chess == 0:
            logger()(f'train start.')
            loader = data_cache.get_loader(conf.batch_size)
            model.train()
            for _ in range(conf.epoch_num):
                loss_record = []
                for bat_state, bat_dist, bat_winner in loader:
                    bat_state, bat_dist, bat_winner = bat_state.to(device), bat_dist.to(device), bat_winner.to(device)
                    optimizer.zero_grad()
                    prob, value = model(bat_state)
                    loss = loss_func(prob, value, bat_dist, bat_winner)
                    loss.backward()
                    optimizer.step()
                    loss_record.append(loss.item())
                logger()(f'train epoch {ep_num} loss: {sum(loss_record) / float(len(loss_record))}')
                ep_num += 1
                if ep_num % conf.checkpoint_save_every_num == 0:
                    save_checkpoint(
                        os.path.join(conf.checkpoint_save_dir, f'epoch_{ep_num}'),
                        ep_num, chess_num, model.state_dict(), optimizer.state_dict(), lr_schedule.state_dict(), data_cache
                    )
            logger()(f'train end.')
        chess_num += 1
        save_chess_record(
            os.path.join(conf.checkpoint_save_dir, f'chess_record_{chess_num}.pkl'),
            chess_record
        )
        # break

    pass
