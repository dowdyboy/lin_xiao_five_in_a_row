import os


class MCTSConfig:
    
    def __init__(self):
        super(MCTSConfig, self).__init__()
        # 策略价值网络权重参数
        self.pretrained_path = None
        # 要生成几盘棋局
        self.num_chess = None
        # 每一步的模拟次数
        self.simulate_count = None
        # 棋局记录的输出目录
        self.save_dir = None
        # 日志文件
        self.log_file = None

    def set_pretrained_path(self, pretrained_path):
        self.pretrained_path = pretrained_path

    def set_chess_config(self, num_chess, simulate_count):
        self.num_chess = num_chess
        self.simulate_count = simulate_count

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    def set_log_file(self, log_file):
        self.log_file = log_file
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

    def __str__(self):
        ret = '[MCTS CONFIG]\n'
        attrs = list(filter(lambda x: not str(x).startswith('__') and not str(x).startswith('set'), dir(self)))
        sorted(attrs)
        for a in attrs:
            ret += '{}: {}\n'.format(a, getattr(self, a))
        return ret
