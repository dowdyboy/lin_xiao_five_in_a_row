import os
import random
import json
import copy
import threading

import torch
from flask import Flask, request, make_response, jsonify, render_template

from mcts import mcts_do_chess
from mcts.monte_tree_v2 import MonteTree
from net import LinXiaoNet

app = Flask(__name__, template_folder='server/templates', static_folder='server/static')
state_result = {}
trees = {}

chess_size = 8
simulate_count = 100
model = LinXiaoNet(3)
device = 'cpu' if torch.cuda.is_available() else 'cpu'

model_pretrain_path = None


def calculate_next_state(state_id, cur_state, player):
    global state_result
    print('start to calculate next state...')
    new_state = copy.deepcopy(cur_state)
    next_x, next_y = mcts_do_chess(model, simulate_count, cur_state, player)
    new_state[next_x][next_y] = player
    state_result[state_id] = new_state
    print('end calculate :', state_result)


def calculate_next_state_v2(state_id, tree_id, pos, cur_state, player):
    global state_result, trees
    print('start to calculate next state...')
    new_state = copy.deepcopy(cur_state)
    tree: MonteTree = trees[tree_id]
    _, (next_x, next_y) = tree.vs_game(pos)
    new_state[next_x][next_y] = player
    state_result[state_id] = new_state
    print('end calculate :\n', state_result)


@app.route('/hello')
def hello_world():
    return 'hello, world'


@app.route('/')
def index():
    return render_template('index.html')


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
