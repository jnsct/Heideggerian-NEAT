import neat
import numpy as np
from multiprocessing import Pool, Value

# generate a group of random board states to be used for training
def generate_board(size, board_num):
    all_boards = []
    for b in range(board_num):
        # position of horizontal wall, which bisects board
        wall_pos = np.random.randint(2, size-2)
        # position of door in wall
        door_pos = np.random.randint(3, size-3)

        # door coordinate
        door = [[wall_pos,door_pos]]

        # pits not generated, to be added later
        pits = []

        # create array with the set board function
        board = set_board(wall_pos,pits,size)

        # randomly place items and agent, and append 90 degree rotations of each
        agent   = [[np.random.randint(wall_pos+1,size-1),np.random.randint(door_pos+1, size-1)]]
        agent  += rotate_coords(agent,size)
        key     = [[np.random.randint(wall_pos+1,size-1),np.random.randint(1, door_pos-1)]]
        key    += rotate_coords(key,size)        
        door   += rotate_coords(door,size)
        goal    = [[np.random.randint(1, wall_pos),np.random.randint(1, size-1)]]
        goal   += rotate_coords(goal,size)

        # append each board and its rotations to final list
        boards = []
        for i in range(4):
            boards.append([board[i],agent[i],key[i],door[i],goal[i]])
        all_boards += boards
    return all_boards

# rotate 2D coordinates in 90 degree increments
def rotate_coords(coordinates, size):
    coordinates = np.array(coordinates)
    center_x = (size - 1) / 2
    center_y = (size - 1) / 2
    points = []
    for a in range(1,4):
        angle = a*90
        rotation_matrix = [[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]]
        rotated_coordinates = np.dot(coordinates - [center_x, center_y], rotation_matrix)[0] + [center_x, center_y]
        points.append(np.round(rotated_coordinates).astype(int).tolist())
    return points

# create a one hot encoded board which represents the placement of the walls and pits
def set_board(wall_pos,pits,size):

    # all zero array of rows
    template = np.zeros(size)
    boards = [template.copy()]
    # set wall row to be all ones in binary
    boards[0][wall_pos] = (2**size)-1

    # mirrored
    board3 = template.copy()
    board3[(size-1)-wall_pos] = (2**size)-1

    # 90 degree rotations
    board2 = np.full(size,1<<wall_pos)
    board4 = np.full(size,1<<((size-1)-wall_pos))
    
    # compile all rotations
    boards += [board2,board3,board4]
    
    return boards
    
# return location of item of interest
def positioner(item, has_key, opened_door, game_coords):
    key,door,goal = game_coords

    if item == 0:
        return key

    # don't show the goal unless the agent has opened the door
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

# print out a visual representation of the board, changing with moves made by agent
def print_board_state(board_state, moves, size):
    board,agent,key,door,goal = board_state
    # empty board
    board_print = np.full((size, size), ' ')

    for r in range(len(board)):
        # horizontal wall
        if board[r] == (2**size)-1:
            horizontal = True
            wall_pos = r
            break
        # vertical wall
        elif board[r] > 0:
            horizontal = False
            wall_pos = int((size-1)-np.log2(board[r]))

    # indicate walls with #
    for i in range(size):
        if horizontal:
            board_print[wall_pos,i] = '#'
        else:
            board_print[i,wall_pos] = '#'
        
    # pits not currently implemented
    #for p in pits:
    #    board_state[*p]   = 'X'

    # set items
    board_print[*goal] = 'G'
    board_print[*key]  = 'K'
    board_print[*door] = 'D'


    # print agent position and item for each move
    for m in moves:
        board_print[*m[0]] = 'A'
        print('Item:', item_name(m[1]))
        print(board_print,'\n','\n')
        board_print[*m[0]] = ' '

# convert output move to change in agent position
def encode_move(move):
    if move == 0:
        return [0,1]
    elif move == 1:
        return [0,-1]
    elif move == 2:
        return [1,0]
    elif move == 3:
        return [-1,0]
    raise Exception("Invalid Move")


# make desired move and detect whether agent has triggered an item on the board
def make_move(item, agent, has_key, opened_door, move_out, board_state, size):

    board,_,key,door,goal = board_state

    # find new agent position
    move = encode_move(move_out)
    next_pos = [agent[0]+move[0], agent[1]+move[1]]

    # used to detect if horizontal or vertical wall configuration
    horizontal = False
    wall_pos = -1
    
    for r in range(len(board)):
        # horizontal
        if board[r] == (2**size)-1:
            horizontal = True
            wall_pos = r
            break
        # vertical
        elif board[r] > 0:
            horizontal = False
            wall_pos = (size-1)-np.log2(board[r])

    # significant agent position to detect wall collision
    if horizontal:
        pos_sig = next_pos[0]
    else:
        pos_sig = next_pos[1]
        
    # iteracting with door
    if list(next_pos) == list(door):
        if has_key and item == 2:
            # opening door
            return next_pos, has_key, 1, 0
        # no key
        return agent, has_key, 0, 0

    
    # bumping into wall
    if pos_sig == wall_pos or size in next_pos or -1 in next_pos:
        return agent, has_key, opened_door, 0

    # picking up key, must be focused on key
    if next_pos == key  and not has_key and item == 0:
        return next_pos, 1, opened_door, 0


    # falling in pit, ignoring pits for now
    #if next_pos in pits:
    #    return agent, has_key, opened_door, -1

    # reaching goal, must be focused on goal
    if next_pos == goal and item == 1:
        return next_pos, has_key, opened_door, 1

    # move with no iteractions or collisions
    return next_pos, has_key, opened_door, 0

# manhattan distance and direction of item
def dist(p1, p2):
    manhattan =  np.array(p1) - np.array(p2)
    if abs(manhattan[0]) > abs(manhattan[1]):
        if manhattan[0] > 0:
            direction = 0
        else:
            direction = 1
    else:
        if manhattan[0] > 0:
            direction = 2
        else:
            direction = 3
    return [np.linalg.norm(manhattan), direction]

# occurances of mode, to guage diversity
def mode_count(arr):
    mode = np.argmax(np.bincount(arr))
    return np.sum(arr == mode)

# find how homogenous runs are in order to reward diverse movement and item focus
def diverse_runs(runs):
    move_modes = []
    item_modes = []
    for i in range(len(runs)):
        items = []
        moves = []
        for j in range(len(runs[i])-1):
            # collect all moves and items
            moves.append((dist(runs[i][j][0], runs[i][j+1][0])[1]))
            items.append(runs[i][j][1])
        # find how many times mode move is made
        item_modes.append(mode_count(items))
        move_modes.append(mode_count(moves))
    # return the average occurance of mode value accross all runs
    return np.mean(move_modes), np.mean(item_modes)

# average number of steps not taken
def get_fitness(fits,moves):

    # score for all completed runs
    fit_sum = np.sum(fits)

    # if no agent has reached goal, then use diversity metric
    if fit_sum == 0:
        
        # encourage novelty by penalizing similarity of movements
        move_std_val, item_std_val = diverse_runs(moves)

        return  -.01 * (move_std_val + item_std_val)
    return int(fit_sum)
    
def genome_fitness(genome, config, boards, max_steps, size):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # item of interest
    # 0 -> key
    # 1 -> goal
    # 2 -> door
    item = 0

    # keep track of fitnesses
    fitnesses = []

    # record movements made during each run
    all_moves = []
    
    # run model on each board
    for b in boards:
        score       = 0    # score for each move
        fitness     = 0    # fitness for run
        agent       = b[1] # default position of agent
        moves       = []   # moves made during run
        has_key     = 0    # whether the key has been collected
        opened_door = 0    # whether the door has been opened
        steps       = 0    # how many moves agent has made

        
        for i in range(max_steps):
            steps += 1

            # save current item of focus to not be overwritten
            curr_item = item
            # position of current item
            item_pos = positioner(item, has_key, opened_door, b[2:])

            # if the agent has not opened the door, it cannot know where the goal is
            if not opened_door and item == 1:
                prox = (-1, -1)
            else:
                prox = dist(item_pos,agent)

            # combine all information for input
            net_input = np.concatenate([b[0], [has_key], prox, agent,[np.random.uniform(0,1)]])
            
            # run the network, get move and next item of interest
            move, item = net.activate(net_input)

            # covert move and item to integers
            move = int(np.floor(move * 3))
            item = int(np.floor(item * 2))
            
            # try to make move, update game state variables
            agent, has_key, opened_door, score = make_move(curr_item, agent, has_key, opened_door, move, b, size)

            # record move
            moves.append((agent,curr_item))
            
            # if goal is reached or agent has fallen into pit, end run
            if score == 1 or score == -1:
                break

        # calculate fitness
        fitnesses.append(max_steps-steps)
        # add moves of run to all runs
        all_moves.append((moves))
    return fitnesses, all_moves

# run evaluation on a genome
def eval_genomes(genome, config, size, board_num, max_steps):
    boards = generate_board(size, board_num)          # generate set of boards
    fitnesses, moves = genome_fitness(                # find fitnesses and runs for each run in genome
        genome, config, boards, max_steps, size) 
    fitness = get_fitness(fitnesses, moves)           # find overall fitness of genome
    return fitness

# evaluate genomes in parallel
class parallel_evaluator(object):
    def __init__(self, num_workers, eval_function, size, board_num, max_steps, timeout=None, maxtasksperchild=None):

        self.eval_function = eval_genomes
        self.timeout       = timeout
        self.pool          = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)
        self.size          = size
        self.board_num     = board_num
        self.max_steps     = max_steps
        self.history       = []

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def get_history(self):
        return self.history
    
    def evaluate(self, genomes, config):

        jobs = []
        # create a job for each genome
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (
                genome, config, self.size, self.board_num, self.max_steps)))

        # fitnesses of all evaluations
        fitnesses = []

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
            fitnesses.append(genome.fitness)

        # keep track of average fitnesses
        self.history.append(np.mean(fitnesses))
