import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle
import os


class ChessDataset(Dataset):

    def __init__(self, chess_dir, nearest_limit=None, logger=print):
        super(ChessDataset, self).__init__()
        self.chess_dir = chess_dir
        self.chess_filename_list = os.listdir(chess_dir)
        if len(self.chess_filename_list) != 0:
            self.chess_filename_list = sorted(self.chess_filename_list, key=lambda x: int(x.split('.')[0]), reverse=True)
            logger(self.chess_filename_list)
            if nearest_limit is not None:
                self.chess_filename_list = self.chess_filename_list[:nearest_limit]
                logger(self.chess_filename_list)
        self.chess_state_list = []
        self.chess_prob_list = []
        self.chess_value_list = []
        for chess_filename in self.chess_filename_list:
            with open(os.path.join(self.chess_dir, chess_filename), 'rb') as f:
                d = pickle.load(f)
                for i in range(0, len(d)-2, 2):
                    self.chess_state_list.append(d[i])
                    self.chess_prob_list.append(d[i+1])
                    self.chess_value_list.append(float(d[-1]))

    def __getitem__(self, idx):
        # print(torch.from_numpy(self.chess_state_list[idx]).type(torch.float))
        # print(torch.unsqueeze(torch.from_numpy(self.chess_prob_list[idx]).type(torch.float), dim=0))
        # print(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(self.chess_value_list[idx]), dim=0), dim=0), dim=0))
        # return self.chess_state_list[idx], self.chess_prob_list[idx], self.chess_value_list[idx]
        return torch.from_numpy(self.chess_state_list[idx]).type(torch.float), torch.unsqueeze(torch.from_numpy(self.chess_prob_list[idx]).type(torch.float), dim=0), torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(self.chess_value_list[idx]), dim=0), dim=0), dim=0)

    def __len__(self):
        return len(self.chess_value_list)

