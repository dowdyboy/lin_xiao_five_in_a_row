import torch
import numpy as np
import copy


class MonteChessTreeConfig:

    PLAYER_BLACK = 1
    PLAYER_WHITE = -1
    EMPTY_POSITION = 0

    def __init__(self):
        self.chess_size = (16, 16)
        self.first_player = MonteChessTreeConfig.PLAYER_BLACK
        self.c_puct = 1.


class MonteChessState:

    def __init__(self, conf: MonteChessTreeConfig):
        self.conf = conf
        self.chess_state = np.zeros((conf.chess_size[0], conf.chess_size[1]), np.int)
        self.current_player = conf.first_player
        # (player, x, y)
        # self.before_steps = []

    def copy(self):
        ret = MonteChessState(self.conf)
        ret.chess_state = copy.deepcopy(self.chess_state)
        ret.current_player = copy.deepcopy(self.current_player)
        # ret.before_steps = copy.deepcopy(self.before_steps)
        return ret

    def switch_player(self):
        assert self.current_player is not None
        self.current_player = MonteChessTreeConfig.PLAYER_WHITE if self.current_player == MonteChessTreeConfig.PLAYER_BLACK else MonteChessTreeConfig.PLAYER_BLACK

    def update_chess_state(self, pos):
        assert self.chess_state is not None
        # assert self.before_steps is not None
        self.chess_state[pos[0], pos[1]] = self.current_player
        # self.before_steps.append(
        #     (self.current_player, pos[0], pos[1])
        # )

    def get_empty_position(self):
        assert self.chess_state is not None
        pos = np.where(self.chess_state == MonteChessTreeConfig.EMPTY_POSITION)
        return list(zip(pos[0], pos[1]))

    def get_empty_position_mask(self):
        mask = self.chess_state.reshape(-1) == MonteChessTreeConfig.EMPTY_POSITION
        return mask

    def get_black_position(self):
        assert self.chess_state is not None
        pos = np.where(self.chess_state == MonteChessTreeConfig.PLAYER_BLACK)
        return list(zip(pos[0], pos[1]))

    def get_white_position(self):
        assert self.chess_state is not None
        pos = np.where(self.chess_state == MonteChessTreeConfig.PLAYER_WHITE)
        return list(zip(pos[0], pos[1]))

    def trans_state_format(self):
        black_state = np.zeros_like(self.chess_state)
        white_state = np.zeros_like(self.chess_state)
        cur_player = np.full(self.chess_state.shape, self.current_player)
        black_state[self.chess_state == MonteChessTreeConfig.PLAYER_BLACK] = MonteChessTreeConfig.PLAYER_BLACK
        white_state[self.chess_state == MonteChessTreeConfig.PLAYER_WHITE] = MonteChessTreeConfig.PLAYER_WHITE
        all_state = np.array([black_state, white_state, cur_player])
        return all_state


class MonteChessTree:

    def __init__(self, conf: MonteChessTreeConfig, model):
        self.conf = conf
        self.model = model
        self.root: MonteChessTreeNode = None

    def reset(self, state=None):
        node = MonteChessTreeNode(self)
        state = MonteChessState(self.conf) if state is None else state
        node.init_state(state)
        node.set_parent(None)
        self.root = node


class MonteChessTreeNode:

    STATE_BLACK_WIN = 1
    STATE_WHITE_WIN = -1
    STATE_DOING = 0
    STATE_TWO_WIN = 2

    def __init__(self, tree: MonteChessTree):
        self.tree = tree
        self.state: MonteChessState = None
        self.N = None
        self.W = None
        self.Q = None
        self.P = None
        self.V = None
        self.children = {}
        self.parent: MonteChessTreeNode = None
        self.node_state = MonteChessTreeNode.STATE_DOING
        self.next_step_cache = None

    def init_state(self, state: MonteChessState):
        self.state = state
        self.N = np.zeros(self.tree.conf.chess_size[0] * self.tree.conf.chess_size[1], dtype=np.int)
        self.W = np.zeros(self.tree.conf.chess_size[0] * self.tree.conf.chess_size[1], dtype=np.float)
        self.Q = np.zeros(self.tree.conf.chess_size[0] * self.tree.conf.chess_size[1], dtype=np.float)
        # self.P = np.zeros(self.tree.conf.chess_size[0] * self.tree.conf.chess_size[1], dtype=np.float)
        prob, v = self.model_pred(state)
        prob = prob.reshape(-1)
        self.P = prob
        self.V = v[0]
        self.node_state = self.charge_node_state()

    def set_parent(self, p):
        self.parent = p

    def model_pred(self, state: MonteChessState):
        trans_state = state.trans_state_format()
        trans_state = trans_state.reshape((1, ) + trans_state.shape)
        trans_state = torch.from_numpy(trans_state).type(torch.float)
        prob_out, value_out = self.tree.model(trans_state)
        prob_out = torch.squeeze(torch.squeeze(prob_out, dim=0), dim=0).detach().numpy()
        value_out = torch.squeeze(torch.squeeze(torch.squeeze(value_out, dim=0), dim=0), dim=0).detach().numpy()
        return prob_out, value_out

    def select_next_position_idx(self):
        select_v = self.Q + self.tree.conf.c_puct * self.P * (np.sqrt(np.sum(self.N)) / (1. + self.N))
        select_v[self.state.get_empty_position_mask() == False] = 0.
        position_idx = np.argmax(select_v)
        return position_idx, (position_idx // self.tree.conf.chess_size[1], position_idx % self.tree.conf.chess_size[1])

    def has_child_node(self, idx):
        return idx in self.children.keys()

    def record_next_step(self, next_step):
        self.next_step_cache = next_step

    def get_next_step_and_clear(self):
        assert self.next_step_cache is not None
        next_step = self.next_step_cache
        self.next_step_cache = None
        return next_step, (next_step // self.tree.conf.chess_size[1], next_step % self.tree.conf.chess_size[1])

    def charge_node_state(self):
        black_pos = self.state.get_black_position()
        white_pos = self.state.get_white_position()
        for cur_x, cur_y in black_pos:

            continuous_count = 0
            for x in range(max(0, cur_x-4), min(self.tree.conf.chess_size[0], cur_x+4+1)):
                if (x, cur_y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_BLACK_WIN

            continuous_count = 0
            for y in range(max(0, cur_y-4), min(self.tree.conf.chess_size[1], cur_y+4+1)):
                if (cur_x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_BLACK_WIN

            continuous_count = 0
            for x, y in zip(range(max(0, cur_x-4), min(self.tree.conf.chess_size[0], cur_x+4+1)), range(max(0, cur_y-4), min(self.tree.conf.chess_size[1], cur_y+4+1))):
                if (x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_BLACK_WIN

            continuous_count = 0
            for x, y in zip(range(cur_x-4, cur_x+4+1), range(cur_y+4, cur_y-4-1, -1)):
                if (x, y) in black_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_BLACK_WIN
            # break
        for cur_x, cur_y in white_pos:
            continuous_count = 0
            for x in range(max(0, cur_x-4), min(self.tree.conf.chess_size[0], cur_x+4+1)):
                if (x, cur_y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_WHITE_WIN

            continuous_count = 0
            for y in range(max(0, cur_y-4), min(self.tree.conf.chess_size[1], cur_y+4+1)):
                if (cur_x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_WHITE_WIN

            continuous_count = 0
            for x, y in zip(range(max(0, cur_x-4), min(self.tree.conf.chess_size[0], cur_x+4+1)), range(max(0, cur_y-4), min(self.tree.conf.chess_size[1], cur_y+4+1))):
                if (x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_WHITE_WIN

            continuous_count = 0
            for x, y in zip(range(cur_x-4, cur_x+4+1), range(cur_y+4, cur_y-4-1, -1)):
                if (x, y) in white_pos:
                    continuous_count += 1
                else:
                    continuous_count = 0
                if continuous_count > 4:
                    return MonteChessTreeNode.STATE_WHITE_WIN
        if (len(white_pos) + len(black_pos)) >= self.tree.conf.chess_size[0] * self.tree.conf.chess_size[1]:
            return MonteChessTreeNode.STATE_TWO_WIN
        return MonteChessTreeNode.STATE_DOING

    def is_over(self):
        return self.node_state == MonteChessTreeNode.STATE_BLACK_WIN or self.node_state == MonteChessTreeNode.STATE_WHITE_WIN or self.node_state == MonteChessTreeNode.STATE_TWO_WIN


def test(model):
    conf = MonteChessTreeConfig()
    tree = MonteChessTree(conf, model)
    tree.reset()

    cur_node = tree.root

    for _ in range(10000):
        is_extend_node = False
        next_idx, next_pos = cur_node.select_next_position_idx()
        if cur_node.has_child_node(next_idx):
            cur_node.record_next_step(next_idx)
            cur_node = cur_node.children[next_idx]
        else:
            is_extend_node = True
            next_state = cur_node.state.copy()
            next_state.update_chess_state(next_pos)
            next_state.switch_player()
            new_node = MonteChessTreeNode(tree)
            new_node.init_state(next_state)
            new_node.set_parent(cur_node)
            cur_node.children[next_idx] = new_node
            cur_node.record_next_step(next_idx)
            cur_node = new_node
        # 判断是否执行回传
        if is_extend_node or cur_node.is_over():
            if cur_node.is_over():
                print('over')
            p_node = cur_node.parent
            while p_node is not None:
                chess_pos, _ = p_node.get_next_step_and_clear()
                v = 0.
                if cur_node.node_state == MonteChessTreeNode.STATE_DOING:
                    v = cur_node.V
                else:
                    v = 1. if cur_node.node_state == MonteChessTreeNode.STATE_BLACK_WIN else -1. if cur_node.node_state == MonteChessTreeNode.STATE_WHITE_WIN else 0.
                p_node.N[chess_pos] = p_node.N[chess_pos] + 1
                p_node.W[chess_pos] = p_node.W[chess_pos] + v
                p_node.Q = p_node.W / (p_node.N + 1e-9)
                p_node = p_node.parent
                cur_node = cur_node.parent
            cur_node = tree.root

    print(cur_node.children)

