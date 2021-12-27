import torch
from torch.utils.data import TensorDataset, DataLoader


# 训练数据缓存池
class TrainDataCache:

    def __init__(self, length=1000, num_worker=0):
        self.state = []
        self.distrib = []
        self.winner = []
        self.length = length
        self.num_worker = num_worker

    def is_empty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.append(item["state"])
        self.distrib.append(item["distribution"])
        self.winner.append(item["value"])
        # 缓存池大小固定，移除旧数据
        if len(self.state) >= self.length:
            self.state = self.state[1:]
            self.distrib = self.distrib[1:]
            self.winner = self.winner[1:]

    def seq(self):
        return self.state, self.distrib, self.winner

    def get_loader(self, batch_size=16):
        state, distrib, winner = self.seq()
        # 将数组转换为匹配网络格式的张量
        tensor_x = torch.stack(tuple([torch.from_numpy(s).to(torch.float32) for s in state]))
        tensor_y1 = torch.stack(tuple([torch.Tensor(y1).view(1, state[0].shape[1], state[0].shape[2]).to(torch.float32) for y1 in distrib]))
        tensor_y2 = torch.stack(tuple([torch.Tensor([float(y2)]).view(1, 1, 1).to(torch.float32) for y2 in winner]))
        # 创建loader
        dataset = TensorDataset(tensor_x, tensor_y1, tensor_y2)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_worker)
        return loader
