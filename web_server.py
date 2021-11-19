import os
import random
import json
import copy
import threading

import torch
from flask import Flask, request, make_response, jsonify, render_template

from mcts import mcts_do_chess
from net import LinXiaoNet

app = Flask(__name__, template_folder='server/templates', static_folder='server/static')
state_result = {}

simulate_count = 2000
model_pretrain_path = None
model = LinXiaoNet(3)
if model_pretrain_path is not None:
    filename_list = os.listdir(model_pretrain_path)
    model_filename = None
    for filename in filename_list:
        if filename.find('model') > -1:
            model_filename = filename
    if model_filename is not None:
        model.load_state_dict(torch.load(os.path.join(model_pretrain_path, model_filename), map_location='cpu'))
    print('successfully load pretrained : {}'.format(model_pretrain_path))


def calculate_next_state(state_id, cur_state, player):
    global state_result
    print('start to calculate next state...')
    new_state = copy.deepcopy(cur_state)
    next_x, next_y = mcts_do_chess(model, simulate_count, cur_state, player)
    new_state[next_x][next_y] = player
    state_result[state_id] = new_state
    print('end calculate :', state_result)


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
    t = threading.Thread(target=calculate_next_state, args=(state_id, req['chess_state'], req['player']))
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


if __name__ == '__main__':
    app.run(
        '0.0.0.0',
        8080
    )
