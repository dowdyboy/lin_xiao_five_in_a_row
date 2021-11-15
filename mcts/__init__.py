import os
import pickle
from mcts.monte_tree import MonteChessTreeConfig, MonteChessTree, MonteChessTreeNode, MonteChessState
from utils.log import MyNetLogger


def mcts_gen_chess(model, num_chess, simulate_count, save_dir, log_file_path=None):
    chess_save_dir = save_dir
    if not os.path.isdir(chess_save_dir):
        os.makedirs(chess_save_dir)

    start_num = 0
    chess_filename_list = os.listdir(chess_save_dir)
    if len(chess_filename_list) != 0:
        start_num = sorted(list(map(lambda x: int(x.split('.')[0]), chess_filename_list)), reverse=True)[0] + 1

    conf = MonteChessTreeConfig()
    tree = MonteChessTree(conf, model)

    logger = print if log_file_path is None else MyNetLogger.default(log_file_path)

    for i in range(start_num, num_chess):
        chess_record = []
        tree.reset()
        while not tree.root.is_over():
            # print(tree.root.state.chess_state)
            logger(tree.root.state.chess_state)
            cur_node = tree.root
            backward_count = 0
            while backward_count < simulate_count:
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

                    v = 0.
                    if cur_node.is_over():
                        # print('simulate chess over')
                        logger('simulate chess over')
                        v = 1. if cur_node.node_state == MonteChessTreeNode.STATE_BLACK_WIN else -1. if cur_node.node_state == MonteChessTreeNode.STATE_WHITE_WIN else 0.
                    else:
                        v = cur_node.V

                    p_node = cur_node.parent
                    while p_node is not None:
                        chess_pos, _ = p_node.get_next_step_and_clear()
                        p_node.N[chess_pos] = p_node.N[chess_pos] + 1
                        p_node.W[chess_pos] = p_node.W[chess_pos] + v
                        p_node.Q = p_node.W / (p_node.N + 1e-9)
                        p_node = p_node.parent
                        cur_node = cur_node.parent
                    cur_node = tree.root
                    backward_count += 1

            saved_state = tree.root.state.trans_state_format()
            next_idx, next_pos, saved_prob = tree.root.get_real_step()
            chess_record.append(saved_state)
            chess_record.append(saved_prob.reshape(conf.chess_size[0], conf.chess_size[1]))

            next_state = tree.root.state.copy()
            next_state.update_chess_state(next_pos)
            next_state.switch_player()
            tree.reset(next_state)

        chess_record.append(tree.root.state.trans_state_format())
        chess_record.append(tree.root.node_state)

        # print(tree.root.state.chess_state)
        # print('no.{} chess finished and saved.'.format(i))
        logger(tree.root.state.chess_state)
        logger('no.{} chess finished and saved.'.format(i))
        with open(os.path.join(chess_save_dir, str(i) + '.pkl'), 'wb+') as f:
            pickle.dump(chess_record, f)
