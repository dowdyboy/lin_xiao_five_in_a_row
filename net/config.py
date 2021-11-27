import os

from net.data.train_data_cache import TrainDataCache

class LinXiaoNetConfig:

    def __init__(self):
        # 是否使用cuda
        self.use_cuda = False
        # 预训练路径
        self.pretrain_path = None
        # 网络输入大小
        self.input_shape = None

        # 训练epoch个数
        self.epoch_num = 0
        # batch size
        self.batch_size = 0
        # 初始学习率
        self.init_lr = 0.

        # 每多少次记录一个检查点
        self.checkpoint_save_every_num = None
        # 检查点存储目录
        self.checkpoint_save_dir = None

        # 日志目录
        self.log_file = None

        # loader的worker个数
        self.num_worker = 0

        # 数据集
        self.dataset_dir = None
        # self.data_cache: TrainDataCache = None

    # 设置网络输入大小
    def set_input_shape(self, w, h):
        if w % 2 != 0 or h % 2 != 0:
            raise ValueError('input shape must % 2')
        self.input_shape = (h, w)

    # 设置是否使用cuda
    def set_cuda(self, use_cuda):
        self.use_cuda = use_cuda

    # 设置训练相关信息
    def set_train_info(self, epoch_num, batch_size, lr):
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.init_lr = lr

    # 设置loader的worker个数
    def set_num_worker(self, num):
        self.num_worker = num

    # 设置检查点配置信息
    def set_checkpoint_config(self, checkpoint_save_every_num, checkpoint_save_dir):
        self.checkpoint_save_every_num = checkpoint_save_every_num
        self.checkpoint_save_dir = checkpoint_save_dir
        if not os.path.isdir(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

    # 设置预训练状态路径
    def set_pretrained_path(self, pretrain_path):
        self.pretrain_path = pretrain_path

    # 配置日志信息
    def set_log(self, log_file):
        self.log_file = log_file
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

    # 配置数据集目录
    def set_dataset_dir(self, dataset_dir):
        self.dataset_dir = dataset_dir

    # 设置数据集缓存对象，用于生成loader
    # def set_data_cache(self, data_cache):
    #     self.data_cache = data_cache

    def __str__(self):
        ret = '[LinXiaoNet CONFIG]\n'
        attrs = list(filter(lambda x: not str(x).startswith('__') and not str(x).startswith('set'), dir(self)))
        sorted(attrs)
        for a in attrs:
            ret += '{}: {}\n'.format(a, getattr(self, a))
        return ret
