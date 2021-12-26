import sys

import torch
import numpy as np

from utils.log import logger

cPrut = 0.1
temperature = 1.


# 将棋盘状态转换为张量
# 转化原则为第一个维度是对手的棋盘布局，第二个维度是自己的棋盘布局，第三个维度是棋手标志
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


# 将二维位置索引转为一维位置索引
def pos_pair2pos_idx(pair, chess_size):
    return pair[0] * chess_size + pair[1]


# 将一维位置索引转为二维位置索引
def pos_idx2pos_pair(pos_idx, chess_size):
    return pos_idx // chess_size, pos_idx % chess_size


# 真实落子位置计算类
class MonteNodeDistCalc:

    def __init__(self, chess_size):
        self.chess_size = chess_size
        self.map = {}
        self.order = []
        self.reset()

    # 重置计算类
    def reset(self):
        self.map = {}
        self.order = []
        for i in range(self.chess_size):
            for j in range(self.chess_size):
                # 将棋盘每个位置的信息进行初始化
                pos_idx = i * self.chess_size + j
                self.order.append(pos_idx)
                self.map[pos_idx] = 0

    def push(self, key, value):
        # 录入所有边界点的一维位置索引和对应的访问次数信息
        self.map[key] = value

    def get(self, train=True):
        result = []
        choice_pool = []
        choice_prob = []
        for key in self.order:
            if self.map[key] != 0:
                # 对于每个位置索引，如果有被访问过，则录入其位置信息，
                # 并录入计算后的访问次数
                choice_pool.append(key)
                tmp = np.float_power(self.map[key], 1 / temperature)
                choice_prob.append(tmp)
                result.append(tmp)
                self.map[key] = 0
            else:
                # 如果没有访问过，则直接置0
                result.append(0)
        # 对访问次数求和
        he = sum(result)
        for i in range(len(result)):
            if result[i]:
                # 计算每个分支访问次数的占比
                result[i] = result[i] / he
        # 计算被访问过的分支访问次数的占比，即将被选择的概率
        choice_prob = [choice/he for choice in choice_prob]
        if train:
            # 如果是训练阶段，则从有被访问过的分支中选择一个，
            # 选择的概率是 依据次数计算出的概率，加上一个迪利克雷分布（引入随机性）
            move = np.random.choice(choice_pool, p=0.8 * np.array(choice_prob) + 0.2 * np.random.dirichlet(0.3*np.ones(len(choice_prob))))
        else:
            # 如果是对弈阶段，则直接选择依据次数计算出的概率的最大分支
            move = choice_pool[np.argmax(choice_prob)]
        return move, result


# 模型封装类
class MonteNetWrapper:

    def __init__(self, model: torch.nn.Module, device):
        super(MonteNetWrapper, self).__init__()
        self.model = model
        self.device = device

    # 根据棋盘状态进行正向传播
    def eval(self, chess_state, single=True):
        self.model.eval()
        chess_state = torch.from_numpy(chess_state).unsqueeze(0).to(self.device).type(torch.float32)
        with torch.no_grad():
            prob, v = self.model(chess_state)
        # 返回评估的落子概率、状态价值
        if single:
            return prob[0, 0, :, :].view(-1).cpu().numpy(), v[0, 0, 0, :].cpu().item()
        else:
            return prob.view(prob.size(0), -1).cpu().numpy(), v.view(v.size(0), -1).cpu().numpy()


# 棋盘状态类
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

    # 重置棋盘状态
    def reset(self, chess_state=None):
        if chess_state is None:
            # 如果没有传递状态矩阵，则重置状态为新棋盘
            self.chess_state = np.zeros((self.chess_size, self.chess_size))
            self.current_player = 1
            self.black_count = 0
            self.white_count = 0
            self.last_step = None
            self.chess_win_state = MonteChessState.CHESS_STATE_DOING
        else:
            # 传递了棋盘状态矩阵，则先更新当前的状态矩阵
            self.chess_state = np.array(chess_state, copy=True)
            # 重新计算当前棋盘状态黑白子的个数
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
            # 根据棋盘中落子的状态来判断当前该谁落子了
            if black_count == white_count:
                self.current_player = 1
            elif black_count == white_count + 1:
                self.current_player = -1
            else:
                raise ValueError('black count and white count error')
            self.last_step = None
            # 检查棋盘状态，判断输赢
            self.chess_win_state = self.check_chess_state()

    # 落子更新棋盘状态
    def step(self, row, col):
        # 在指定位置落子
        self.chess_state[row, col] = self.current_player
        if self.current_player == 1:
            self.black_count += 1
        elif self.current_player == -1:
            self.white_count += 1
        else:
            raise ValueError('current player value error')
        # 更新下次落子棋手
        self.current_player = -self.current_player
        self.last_step = [row, col]
        # 每次完成落子后，检查棋盘状态
        self.chess_win_state = self.check_chess_state()

    # 获取棋盘中所有黑子的二维位置索引
    def get_black_position(self):
        pos = np.where(self.chess_state == 1)
        return list(zip(pos[0], pos[1]))

    # 获取棋盘中所有白子的二维位置索引
    def get_white_position(self):
        pos = np.where(self.chess_state == -1)
        return list(zip(pos[0], pos[1]))

    # 检查棋盘状态判断输赢
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
        # 父节点，是边节点
        self.parent = parent
        # 节点代表的选手
        self.player = player
        # 当前节点被访问的次数
        self.count = 0.
        # 边子节点
        self.children = {}

    # 拓展边
    def expand_edge(self, pos_idx, prob):
        self.children[pos_idx] = MonteTreeEdge(self, pos_idx, prob)

    # 通过位置索引获取一个子节点
    def get_child(self, pos_idx):
        return self.children[pos_idx].child_node

    # 反传更新参数
    def backup(self, v):
        # 当前节点访问次数+1
        self.count += 1.
        if self.parent is not None:
            self.parent.backup(v)

    # 判断当前节点是否拓展过
    def is_expand(self):
        return len(self.children) > 0

    # 通过ucb值选择下一个节点
    def ucb_select(self):
        ucb_max = -sys.maxsize
        ucb_max_pos_idx = None
        for pos_idx in self.children.keys():
            if self.children[pos_idx].ucb() > ucb_max:
                ucb_max = self.children[pos_idx].ucb()
                ucb_max_pos_idx = pos_idx
        is_expand = False
        # 如果所选择的边节点没有后续节点，则进行拓展
        if not self.children[ucb_max_pos_idx].is_expand():
            self.children[ucb_max_pos_idx].expand_node()
            is_expand = True
        # 所选边节点的后续节点作为被选节点
        select_node = self.children[ucb_max_pos_idx].child_node
        return select_node, is_expand, self.children[ucb_max_pos_idx].pos_idx

    # 获取当前节点的真实落子
    def get_distribute(self, dist_calc: MonteNodeDistCalc, is_train):
        # 重置计算类
        dist_calc.reset()
        for pos_idx in self.children.keys():
            # 将当前节点所有的边节点的一维位置信息、访问次数信息输入
            dist_calc.push(pos_idx, self.children[pos_idx].count)
        # 获取最终落子的一维位置索引
        return dist_calc.get(is_train)


class MonteTreeEdge:

    def __init__(self, parent, pos_idx, prob):
        super(MonteTreeEdge, self).__init__()
        # 父节点
        self.parent = parent
        # 当前边节点的得分，由策略网络给出
        self.prob = prob
        # 边节点的一维索引编号
        self.pos_idx = pos_idx
        # 该边访问次数
        self.count = 1.
        # 边节点所指的下一个节点
        self.child_node = None
        # 该边的奖励价值
        self.v = 0.

    # 通过ucb公式计算当前边节点的 真实奖励 用于路径选择
    def ucb(self):
        # 计算当前边节点的平均价值
        q = self.v / self.count
        # 平均价值 + 一个调整值（一个权重 * 当前节点的策略价值 * 当前层所有路径的总次数/当前路径次数）
        # 这个调整值表示，在刚开始时候当前路径次数小，调整值就大，就更可能访问当前路径
        # 也就是说，会在游戏刚开始时候更可能访问没有被访问过的路径
        return q + cPrut * self.prob * np.sqrt(self.parent.count) / (1 + self.count)

    # 回传更新参数
    def backup(self, v):
        # 访问次数+1
        self.count += 1.
        # 累加价值
        self.v += v
        self.parent.backup(-v)

    # 拓展节点
    def expand_node(self):
        # 选手要易位
        self.child_node = MonteTreeNode(self, -self.parent.player)

    # 判断是否拓展了节点
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

    # 根据落子位置索引，更新树
    def step_update(self, pos_idx):
        # 如果落子位置的边节点没有节点，则拓展新的节点
        if not self.current_node.children[pos_idx].is_expand():
            self.current_node.children[pos_idx].expand_node()
        # 拿到落子位置所对应的节点
        next_node = self.current_node.children[pos_idx].child_node
        # 将其父节点置空，相当于这个节点要做根节点了
        next_node.parent = None
        return next_node

    # 进行蒙特卡洛模拟
    def simulate(self):
        for _ in range(self.simulate_count):
            # 获取当前节点
            cur_node = self.current_node
            # 重置模拟状态
            self.simulate_state.reset(self.real_state.chess_state)
            state = self.simulate_state.chess_state
            is_continue, is_expand = True, False
            while is_continue and not is_expand:
                # 如果当前节点没有拓展过，则进行边节点的拓展
                if not cur_node.is_expand():
                    # 根据策略网络的评估拿到落子概率
                    prob, _ = self.net.eval(transfer_to_net_input(state, self.simulate_state.current_player, self.chess_size))
                    # 拿到当前状态还没有落子的二维位置索引
                    valid_pos = list(np.argwhere(state == 0))
                    for pos in valid_pos:
                        # 对全部没有落子的位置进行边节点拓展
                        cur_node.expand_edge(pos_pair2pos_idx(pos, self.chess_size), prob[pos_pair2pos_idx(pos, self.chess_size)])
                # 根据ucb值选择下一个落子的节点
                cur_node, is_expand, pos_idx = cur_node.ucb_select()
                # 根据落子位置更新棋盘状态
                self.simulate_state.step(pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])
                # 当前状态指针变更
                state = self.simulate_state.chess_state
                # 判断棋局是否结束，结束了的话就设置标记
                if self.simulate_state.chess_win_state != MonteChessState.CHESS_STATE_DOING:
                    is_continue = False
            if not is_continue:
                # TODO: 此处不管胜负都回传1，为什么不是黑子胜了回传1、白子回传-1呢？
                # 如果棋局结束，则由环境给出回传价值；针对上面问题的答案，因为如果当前棋局已经结束，则最后落子的棋子一定是赢家，
                # 这样，不管是黑棋赢了还是白棋赢了，则都应该给最后一个落子回传正值价值，给倒数第二个棋子回传负值价值，依此类推
                cur_node.backup(1.)
            elif is_expand:
                # 如果拓展了新的节点，则通过策略网络获取评估价值
                _, v = self.net.eval(transfer_to_net_input(state, self.simulate_state.current_player, self.chess_size))
                # 回传当前棋盘状态的价值
                cur_node.backup(v)

    def self_game(self):
        is_continue = True
        step_record = []
        while is_continue:
            self.simulate()
            pos_idx, distribute = self.current_node.get_distribute(self.dist_calc, True)
            self.real_state.step(pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])
            logger()(f'\n({pos_idx2pos_pair(pos_idx, self.chess_size)[0]},{pos_idx2pos_pair(pos_idx, self.chess_size)[1]})\n{str(self.real_state.chess_state)}')
            self.current_node = self.step_update(pos_idx)
            step_record.append(
                (distribute, pos_idx)
            )
            if self.real_state.chess_win_state != MonteChessState.CHESS_STATE_DOING:
                is_continue = False
        # logger()(f'\n{str(self.real_state.chess_state)}')
        logger()(f'winner: {self.real_state.chess_win_state}, black_count: {self.real_state.black_count}, white_count: {self.real_state.white_count}')
        self.reset()
        return step_record

    def vs_game(self, pos_pair):
        # 更新棋盘状态位置
        self.real_state.step(pos_pair[0], pos_pair[1])
        # 如果当前人类落子在树中没有对应的节点，则为其先创建边节点；边节点的概率设置为1，因为这是人类的真实落子
        if pos_pair2pos_idx(pos_pair, self.chess_size) not in self.current_node.children.keys():
            self.current_node.children[pos_pair2pos_idx(pos_pair, self.chess_size)] = MonteTreeEdge(self.current_node, pos_pair2pos_idx(pos_pair, self.chess_size), 1.)
        # 根据当前人类落子的位置，更新树（步进树的根节点），重置根节点
        self.current_node = self.step_update(pos_pair2pos_idx(pos_pair, self.chess_size))
        # 根据新的状态进行模拟
        self.simulate()
        # 获取最终落子位置
        pos_idx, distribute = self.current_node.get_distribute(self.dist_calc, False)
        # 更新棋盘状态
        self.real_state.step(pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])
        # 更新树
        self.current_node = self.step_update(pos_idx)
        return pos_idx, (pos_idx2pos_pair(pos_idx, self.chess_size)[0], pos_idx2pos_pair(pos_idx, self.chess_size)[1])












