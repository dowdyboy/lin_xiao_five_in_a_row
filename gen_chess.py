import torch

from mcts.config import MCTSConfig
from mcts import mcts_gen_chess

from net import LinXiaoNet

from utils.log import MyNetLogger

import os

if __name__ == '__main__':
    conf = MCTSConfig()
    conf.set_chess_config(100, 10000)
    conf.set_save_dir('out/m10000_1')
    conf.set_log_file('log/m10000_1.log')
    conf.set_pretrained_path(None)

    logger = print if conf.log_file is None else MyNetLogger.default(conf.log_file)
    logger(conf)

    model = LinXiaoNet(3)

    if conf.pretrained_path is not None:
        filename_list = os.listdir(conf.pretrained_path)
        model_filename = None
        for filename in filename_list:
            if filename.find('model') > -1:
                model_filename = filename
        if model_filename is not None:
            model.load_state_dict(torch.load(os.path.join(conf.pretrained_path, model_filename)))
        logger('successfully load pretrained : {}'.format(conf.pretrained_path))

    mcts_gen_chess(model, conf.num_chess, conf.simulate_count, conf.save_dir, conf.log_file)
