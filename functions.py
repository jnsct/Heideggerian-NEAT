import os
import neat
import visualize
import numpy as np
import pickle
import matplotlib as plt

import functions as f

from multiprocessing import Pool, Value


def rotate_coords(coordinates, dim):
    coordinates = np.array(coordinates)
    center_x = (dim - 1) / 2
    center_y = (dim - 1) / 2
    points = []
    for a in range(1,4):
        angle = a*90
        rotation_matrix = [[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]]
        rotated_coordinates = np.dot(coordinates - [center_x, center_y], rotation_matrix)[0] + [center_x, center_y]
        points.append(np.round(rotated_coordinates).astype(int).tolist())
    return points

# create a one hot encoded board which represents the placement of the walls and pits
def set_board(wall_pos,pits,rotate,dim):

    template = np.zeros(dim)
    boards = [template.copy()]
    boards[0][wall_pos] = (2**dim)-1

    if rotate:
        # mirrored
        board3 = template.copy()
        board3[(dim-1)-wall_pos] = (2**dim)-1

        # 90 degree rotations
        board2 = np.full(dim,1<<wall_pos)

        board4 = np.full(dim,1<<((dim-1)-wall_pos))
        
        boards += [board2,board3,board4]
    
    return boards

def generate_board(dim, rotate, board_num):
    all_boards = []
    for b in range(board_num):
        wall_pos = np.random.randint(2, dim-2)
        #door_pos = np.random.randint(2, dim-2)
        door_pos = np.random.randint(3, dim-3)
        
        door = [[wall_pos,door_pos]]
    
        # ignoring pits, for now
        pits = []

        board = set_board(wall_pos,pits,rotate,dim)

        player  = [[np.random.randint(wall_pos+1,dim-1),np.random.randint(door_pos+1, dim-1)]]
        player += rotate_coords(player,dim)
        key     = [[np.random.randint(wall_pos+1,dim-1),np.random.randint(1, door_pos-1)]]
        key    += rotate_coords(key,dim)        
        door   += rotate_coords(door,dim)
        goal    = [[np.random.randint(1, wall_pos),np.random.randint(1, dim-1)]]
        goal   += rotate_coords(goal,dim)


        #key    = [np.random.randint(wall_pos+1,dim-1),np.random.randint(1, dim-1)]
        #player = [np.random.randint(wall_pos+1,dim-1),np.random.randint(1, dim-1)]

        # inefficient, stack later
        boards = []
        for i in range(4):
            boards.append([board[i],player[i],key[i],door[i],goal[i]])
        all_boards += boards
    return all_boards

# find euclidean distance direction of item
def dist(p1, p2):
    manhattan =  np.array(p1) - np.array(p2)
    if abs(manhattan[0]) > abs(manhattan[1]):
        if manhattan[0] > 0:
            direction = 0.
        else:
            direction = .25
    else:
        if manhattan[0] > 0:
            direction = .75
        else:
            direction = 1.
    return [np.linalg.norm(manhattan), direction]
    
# return location of item of interest
def positioner(item, has_key, opened_door, game_coords):
    key,door,goal = game_coords

    if item == 0:
        return key
        
    elif item == 1:
        if opened_door:
            return goal
        return [-1,-1]

    if item == 2:
        return door
            
    raise Exception("Invalid Item")

# return name of item of interest
def item_name(item):
    if item == 0:
        return "key"
    elif item == 1:
        return "goal"
    elif item == 2:
        return "door"    

    raise Exception("Invalid Item")


def print_board_state(board_state, moves):
    board,player,key,door,goal = board_state

    board_print = np.full((dim, dim), ' ')

    for r in range(len(board)):
        # horizontal
        if board[r] == (2**dim)-1:
            horizontal = True
            wall_pos = r
            break
        # vertical
        elif board[r] > 0:
            horizontal = False
            wall_pos = int((dim-1)-np.log2(board[r]))

    for i in range(dim):
        if horizontal:
            board_print[wall_pos,i] = '#'
        else:
            board_print[i,wall_pos] = '#'
        
    # pits not currently implemented
    #for p in pits:
    #    board_state[*p]   = 'X'

    board_print[*goal] = 'G'
    board_print[*key]  = 'K'
    board_print[*door] = 'D'


    for m in moves:
        board_print[*m[0]] = 'P'
        print('Item:', item_name(m[1]))
        print(board_print,'\n','\n')
        board_print[*m[0]] = ' '

def encode_move(move):
    if move <= .25:
        return [1,0]
    elif move <= .5:
        return [0,-1]
    elif move <= .75:
        return [0,1]
    elif move <= 1:
        return [-1,0]
    raise Exception("Invalid Move")

# detect whether player has triggered an item on the board
def make_move(item, player, has_key, opened_door, move_out, board_state, dim):

    board,_,key,door,goal = board_state
    
    move = encode_move(move_out)
    next_pos = [player[0]+move[0], player[1]+move[1]]

    horizontal = False
    wall_pos = -1
    
    for r in range(len(board)):
        # horizontal
        if board[r] == (2**dim)-1:
            horizontal = True
            wall_pos = r
            break
        # vertical
        elif board[r] > 0:
            horizontal = False
            wall_pos = (dim-1)-np.log2(board[r])

    # significant player position, whether the wall is horizontal or not
    if horizontal:
        pos_sig = next_pos[0]
    else:
        pos_sig = next_pos[1]
        
    # no key to door
    if list(next_pos) == list(door):
        if has_key and (np.random.randint(0,9) > 10 or item == 2):
            # opening door
            #print('opening door')
            return next_pos, has_key, 1, 0
        # no keys to door
        #print('no key')
        return player, has_key, 0, 0

    
    # if bumping into wall
    if pos_sig == wall_pos or dim in next_pos or -1 in next_pos:
        #print('off grid')
        return player, has_key, opened_door, 0

    # picking up key, must be focused on key
    if next_pos == key  and not has_key and (np.random.randint(0,9) > 10 or item == 0):
        return next_pos, 1, opened_door, 0


    # falling in pit, ignoring pits for now
    #if next_pos in pits:
    #    return player, has_key, opened_door, -1

    # reaching goal, must be focused on goal
    if next_pos == goal and (np.random.randint(0,9) > 10 or item == 1):
        #print('reached goal')
        return next_pos, has_key, opened_door, 1
    #print('normal move', player, next_pos)
    return next_pos, has_key, opened_door, 0


def dist(p1, p2):
    manhattan =  np.array(p1) - np.array(p2)
    if abs(manhattan[0]) > abs(manhattan[1]):
        if manhattan[0] > 0:
            direction = 0.
        else:
            direction = .25
    else:
        if manhattan[0] > 0:
            direction = .75
        else:
            direction = 1.
    return [np.linalg.norm(manhattan), direction]

def high_mode(arr):
    mode = np.argmax(np.bincount(arr))
    return np.sum(arr == mode)


# room for efficiency
def diverse_plays(plays):
    move_modes = []
    item_modes = []
    for i in range(len(plays)):
        items = []
        moves = []
        for j in range(len(plays[i])-1):
            moves.append((dist(plays[i][j][0], plays[i][j+1][0])[1])/4)
            items.append(plays[i][j][1]/3)
        item_modes.append(high_mode(items))
        move_modes.append(high_mode(moves))
    return np.mean(move_modes), np.mean(item_modes)


# average number of steps not taken
def get_fitness(fits,plays):

    # encourage novelty by penalizing similarity to the mean of standard deviations of moves and items
    move_std_val, item_std_val = diverse_plays(plays)
    #print(move_std_val)
    #cease

    #with num_entries.get_lock():
    #    num_entries.value += 1
    #    with item_std.get_lock():
    #        item_std.value += item_std_val/num_entries.value
    #    with move_std.get_lock():
    #        move_std.value += move_std_val/num_entries.value
    
    return np.sum(fits) - (move_std_val + item_std_val)*.1
    
# average number of steps taken
def get_eval_fitness(fits):
    return max_steps - np.mean(fits)