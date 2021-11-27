import sys

import torch
import numpy as np

from utils.log import logger

cPrut = 0.1
temperature = 1.


def transfer_to_net_input(chess_state, player, chess_size):
    if player == 1:
        player_layer = np.ones((chess_size, chess_size))
        self_layer = np.array(chess_state > 0).astype(np.float)
        other_layer = np.array(chess_state < 0).astype(np.float)
    else:
        player_layer = np.zeros((chess_size, chess_size))
        self_layer = np.array(chess_state < 0).astype(np.float)
        other_layer = np.array(chess_state > 0).astype(np.float)
    return np.stack([other_layer, self_layer, player_layer])


def pos_pair2pos_idx(pair, chess_size):
    return pair[0] * chess_size + pair[1]


def pos_idx2pos_pair(pos_idx, chess_size):
    return pos_idx // chess_size, pos_idx % chess_size


class MonteNodeDistCalc:

    def __init__(self, chess_size):
        self.chess_size = chess_size
        self.map = {}
        self.order = []
        self.reset()

    def reset(self):
        self.map = {}
        self.order = []
        for i in range(self.chess_size):
            for j in range(self.chess_size):
                pos_idx = i * self.chess_size + j
                self.order.append(pos_idx)
                self.map[pos_idx] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self, train=True):
        result = []
        choice_pool = []
        choice_prob = []
        for key in self.order:
            if self.map[key] != 0:
                choice_pool.append(key)
                tmp = np.float_power(self.map[key], 1 / temperature)
                choice_prob.append(tmp)
                result.append(tmp)
                self.map[key] = 0
            else:
                result.append(0)

        he = sum(result)
        for i in range(len(result)):
            if result[i]:
                result[i] = result[i] / he
        choice_prob = [choice/he for choice in choice_prob]
        if train:
            move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(0.3*np.ones(len(choice_prob))))
        else:
            move = choice_pool[np.argmax(choice_prob)]
        return move, result


class MonteNetWrapper:

    def __init__(self, model: torch.nn.Module, device):
        super(MonteNetWrapper, self).__init__()
        self.model = model
        self.device = device

    def eval(self, chess_state, single=True):
        self.model.eval()
        chess_state = torch.from_numpy(chess_state).unsqueeze(0).to(self.device).type(torch.float32)
        with torch.no_grad():
            prob, v = self.model(chess_state)
        if single:
            return prob[0, 0, :, :].view(-1).cpu().numpy(), v[0, 0, 0, :].cpu().item()
        else:
            return prob.view(prob.size(0), -1).cpu().numpy(), v.view(v.size(0), -1).cpu().numpy()


class MonteChessState:

    CHESS_STATE_DOING = 0
    CHESS_STATE_BLACK_WIN = 1
    CHESS_STATE_WHITE_WIN = -1
    CHESS_STATE_TWO_WIN = 2

    def __init__(self, chess_size):
        super(MonteChessState, self).__init__()
        self.chess_size = chess_size
        self.chess_state = np.zeros((chess_size, chess_size))
        self.current_player = 1
        self.last_step = None
        self.black_count = 0
        self.white_count = 0
        self.chess_win_state = MonteChessState.CHESS_STATE_DOING

    def reset(self, chess_state=None):
        if chess_state is None:
            self.chess_state = np.zeros((self.chess_size, self.chess_size))
            self.current_player = 1
            self.black_count = 0
            self.white_count = 0
            self.last_step = None
            self.chess_win_state = MonteChessState.CHESS_STATE_DOING
        else:
            self.chess_state = np.array(chess_state, copy=True)
            black_count = 0
            white_count = 0
            for row in range(len(self.chess_state)):
                for col in range(len(self.chess_state[0])):
                    if self.chess_state[row][col] == 1:
                        black_count += 1
                    elif self.chess_state[row][col] == -1:
                        white_count += 1
                    elif self.chess_state[row][col] != 0:
                        raise ValueError('self.chess_state[row][col] != 0')
            self.black_count = black_count
            self.white_count = white_count
            if black_count == white_count:
                self.current_player = 1
            elif black_count == white_count + 1:
                self.current_player = -1
            else:
                raise ValueError('black count and white count error')
            self.last_step = None
            self.chess_win_state = self.check_chess_state()

    def step(self, row, col):
        self.chess_state[row, col] = self.current_player
        if self.current_player == 1:
            self.black_count += 1
        elif self.current_player == -1:
            self.white_count += 1
        else:
            raise ValueError('current player value error')
        self.current_player = -self.current_player
        self.last_step = [row, col]
        self.chess_win_state = self.check_chess_state()

    def get_black_position(self):
        pos = np.where(self.chess_state == 1)
        return list(zip(pos[0], pos[1]))

    def get_white_position(self):
        pos = np.where(self.chess_state == -1)
        return list(zip(pos[0], pos[1]))

    def check_chess_state(self):
        black_pos = self.get_black_position()
        white_pos = self.get_white_position()
        for cur_x, cur_y in black_pos:
            continuous_count = 0
            for x in range(max(0, cur_x-4), min(self.chess_state.shape[0], cur_x+4+1)):
                if (x, cur_y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_BLACK_WIN

            continuous_count = 0
            for y in range(max(0, cur_y-4), min(self.chess_state.shape[1], cur_y+4+1)):
                if (cur_x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_BLACK_WIN

            continuous_count = 0
            for x, y in zip(range(max(0, cur_x-4), min(self.chess_state.shape[0], cur_x+4+1)), range(max(0, cur_y-4), min(self.chess_state.shape[1], cur_y+4+1))):
                if (x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_BLACK_WIN

            continuous_count = 0
            for x, y in zip(range(cur_x-4, cur_x+4+1), range(cur_y+4, cur_y-4-1, -1)):
                if (x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_BLACK_WIN
            # break
        for cur_x, cur_y in white_pos:
            continuous_count = 0
            for x in range(max(0, cur_x-4), min(self.chess_state.shape[0], cur_x+4+1)):
                if (x, cur_y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_WHITE_WIN

            continuous_count = 0
            for y in range(max(0, cur_y-4), min(self.chess_state.shape[1], cur_y+4+1)):
                if (cur_x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_WHITE_WIN

            continuous_count = 0
            for x, y in zip(range(max(0, cur_x-4), min(self.chess_state.shape[0], cur_x+4+1)), range(max(0, cur_y-4), min(self.chess_state.shape[1], cur_y+4+1))):
                if (x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_WHITE_WIN

            continuous_count = 0
            for x, y in zip(range(cur_x-4, cur_x+4+1), range(cur_y+4, cur_y-4-1, -1)):
                if (x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessState.CHESS_STATE_WHITE_WIN
        if (len(white_pos) + len(black_pos)) >= self.chess_state.shape[0] * self.chess_state.shape[1]:
            return MonteChessState.CHESS_STATE_TWO_WIN
        return MonteChessState.CHESS_STATE_DOING


class MonteTreeNode:

    def __init__(self, parent, player):
        super(MonteTreeNode, self).__init__()
        self.parent = parent
        self.player = player
        self.count = 0.
        self.children = {}

    def expand_edge(self, pos_idx, prob):
        self.children[pos_idx] = MonteTreeEdge(self, pos_idx, prob)

    def get_child(self, pos_idx):
        return self.children[pos_idx].child_node

    def backup(self, v):
        self.count += 1.
        if self.parent is not None:
            self.parent.backup(v)

    def is_expand(self):
        return len(self.children) > 0

    def ucb_select(self):
        ucb_max = -sys.maxsize
        ucb_max_pos_idx = None
        for pos_idx in self.children.keys():
            if self.children[pos_idx].ucb() > ucb_max:
                ucb_max = self.children[pos_idx].ucb()
                ucb_max_pos_idx = pos_idx
        is_expand = False
        if not self.children[ucb_max_pos_idx].is_expand():
            self.children[ucb_max_pos_idx].expand_node()
            is_expand = True
        select_node = self.children[ucb_max_pos_idx].child_node
        return select_node, is_expand, self.children[ucb_max_pos_idx].pos_idx

    def get_distribute(self, dist_calc: MonteNodeDistCalc, is_train):
        dist_calc.reset()
        for pos_idx in self.children.keys():
            dist_calc.push(pos_idx, self.children[pos_idx].count)
        return dist_calc.get(is_train)


class MonteTreeEdge:

    def __init__(self, parent, pos_idx, prob):
        super(MonteTreeEdge, self).__init__()
        self.parent = parent
        self.prob = prob
        self.pos_idx = pos_idx
        self.count = 1.
        self.child_node = None
        self.v = 0.

    def ucb(self):
        q = self.v / self.count
        return q + cPrut * self.prob * np.sqrt(self.parent.count) / (1 + self.count)

    def backup(self, v):
        self.count += 1.
        self.v += v
        self.parent.backup(-v)

    def expand_node(self):
        self.child_node = MonteTreeNode(self, -self.parent.player)

    def is_expand(self):
        return self.child_node is not None


class MonteTree:

    def __init__(self, model, device, chess_size=16, simulate_count=1000):
        super(MonteTree, self).__init__()
        self.real_state = MonteChessState(chess_size)
        self.simulate_state = MonteChessState(chess_size)
        # self.model = model
        self.net = MonteNetWrapper(model, device)
        self.chess_size = chess_size
        self.simulate_count = simulate_count
        self.current_node = MonteTreeNode(None, 1)
        self.dist_calc = MonteNodeDistCalc(chess_size)

    def reset(self):
        self.current_node = MonteTreeNode(None, 1)
        self.real_state.reset()

    def step_update(self, pos_idx):
        if not self.current_node.children[pos_idx].is_expand():
            self.current_node.children[pos_idx].expand_node()
        next_node = self.current_node.children[pos_idx].child_node
        next_node.parent = None
        return next_node

    def simulate(self):
        for _ in range(self.simulate_count):
            cur_node = self.current_node
            self.simulate_state.reset(self.real_state.chess_state)
            state = self.simulate_state.chess_state
            is_continue, is_expand = True, False
            while is_continue and not is_expand:
                if not cur_node.is_expand():
                    prob, _ = self.net.eval(transfer_to_net_input(state, self.simulate_state.current_player, self.chess_size))
                    valid_pos = list(np.argwhere(state == 0))
                    for pos in valid_pos:
                        cur_node.expand_edge(pos_pair2pos_idx(pos, self.chess_size), prob[pos_pair2pos_idx(pos, self.chess_size)])
                cur_node, is_expand, pos_idx = cur_node.ucb_select()
                self.simulate_state.step(pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])
                state = self.simulate_state.chess_state
                if self.simulate_state.chess_win_state != MonteChessState.CHESS_STATE_DOING:
                    is_continue = False
            if not is_continue:
                # TODO: 此处不管胜负都回传1，为什么不是黑子胜了回传1、白子回传-1呢？
                cur_node.backup(1.)
            elif is_expand:
                _, v = self.net.eval(transfer_to_net_input(state, self.simulate_state.current_player, self.chess_size))
                cur_node.backup(v)

    def self_game(self):
        is_continue = True
        step_record = []
        while is_continue:
            self.simulate()
            pos_idx, distribute = self.current_node.get_distribute(self.dist_calc, True)
            self.real_state.step(pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])
            self.current_node = self.step_update(pos_idx)
            step_record.append(
                (distribute, pos_idx)
            )
            if self.real_state.chess_win_state != MonteChessState.CHESS_STATE_DOING:
                is_continue = False
        logger()(f'\n{str(self.real_state.chess_state)}')
        logger()(f'winner: {self.real_state.chess_win_state}, black_count: {self.real_state.black_count}, white_count: {self.real_state.white_count}')
        self.reset()
        return step_record










