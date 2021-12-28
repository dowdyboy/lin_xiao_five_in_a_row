import os
import random
import json
import copy
import threading

import torch
from flask import Flask, request, make_response, jsonify, render_template

# from mcts import mcts_do_chess
from mcts.monte_tree_v2 import MonteTree
from net import LinXiaoNet
from network import TransplantNet

app = Flask(__name__, template_folder='server/templates', static_folder='server/static')
# 暂存机器落子状态
state_result = {}
# 存放游戏创建的蒙特卡洛树
trees = {}

# 游戏配置
# 棋盘大小
chess_size = 8
# 每次实际落子前，模拟推演的次数
simulate_count = 5000
# 创建决策网络
# model = LinXiaoNet(3)
model = TransplantNet('model_5400.pkl')
device = 'cpu' if torch.cuda.is_available() else 'cpu'
# 加载训练权重
# model_pretrain_path = 'checkpoints/v2train/epoch_655'
model_pretrain_path = None

# 加载权重
if model_pretrain_path is not None:
    model.load_state_dict(torch.load(os.path.join(model_pretrain_path, 'model.pth')))
    print('successful load pretrained model.')

# def calculate_next_state(state_id, cur_state, player):
#     global state_result
#     print('start to calculate next state...')
#     new_state = copy.deepcopy(cur_state)
#     next_x, next_y = mcts_do_chess(model, simulate_count, cur_state, player)
#     new_state[next_x][next_y] = player
#     state_result[state_id] = new_state
#     print('end calculate :', state_result)


def calculate_next_state_v2(state_id, tree_id, pos, cur_state, player):
    global state_result, trees
    print('start to calculate next state...')
    new_state = copy.deepcopy(cur_state)
    tree: MonteTree = trees[tree_id]
    # 讲用户的落子位置传入函数，通过蒙特树进行计算，获取机器落子
    _, (next_x, next_y) = tree.vs_game(pos)
    # 更新棋盘状态
    new_state[next_x][next_y] = player
    # 记录新状态
    state_result[state_id] = new_state
    print('end calculate :\n', state_result)


@app.route('/hello')
def hello_world():
    return 'hello, world'


@app.route('/')
def index():
    return render_template('index.html')


# 接收用户落子
@app.route('/state/put', methods=['POST'])
def put_state():
    req = request.get_json()
    print(type(req), req)
    state_id = random.randint(1000, 100000)
    t = threading.Thread(target=calculate_next_state_v2, args=(state_id, req['tree_id'], req['pos'], req['chess_state'], req['player']))
    t.start()
    ret = {
        'code': 0,
        'msg': 'OK',
        'data': {
            'state_id': state_id
        }
    }
    return jsonify(ret)


# 用户查询机器落子状态
@app.route('/state/get/<state_id>', methods=['GET'])
def get_state(state_id):
    global state_result
    state_id = int(state_id)
    state = 0
    chess_state = None
    if state_id in state_result.keys() and state_result[state_id] is not None:
        state = 1
        chess_state = state_result[state_id]
        state_result[state_id] = None
    ret = {
        'code': 0,
        'msg': 'OK',
        'data': {
            'state': state,
            'chess_state': chess_state
        }
    }
    return jsonify(ret)


# 游戏开始，为这场游戏创建蒙特卡洛树
@app.route('/game/start', methods=['POST'])
def game_start():
    global trees
    global model, device, chess_size, simulate_count
    tree_id = random.randint(1000, 100000)
    trees[tree_id] = MonteTree(model, device, chess_size=chess_size, simulate_count=simulate_count)
    ret = {
        'code': 0,
        'msg': 'OK',
        'data': {
            'tree_id': tree_id
        }
    }
    return jsonify(ret)


# 游戏结束，销毁蒙特卡洛树
@app.route('/game/end/<tree_id>', methods=['POST'])
def game_end(tree_id):
    global trees
    tree_id = int(tree_id)
    trees[tree_id] = None
    ret = {
        'code': 0,
        'msg': 'OK',
        'data': {}
    }
    return ret


if __name__ == '__main__':
    app.run(
        '0.0.0.0',
        8080
    )
