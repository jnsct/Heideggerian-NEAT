import os
import neat
import visualize
import numpy as np
import pickle
import matplotlib as plt

import functions as f

from multiprocessing import Pool, Value

dim = 7
history = []
board_num = 3
max_steps = 50
rotate = True

def genome_fitness(genome, config, boards):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # item of interest
    # 0 -> key
    # 1 -> goal
    # 2 -> door
    item = 0

    # score, getting goal is 1, falling in pit is -1
    score = 0

    fitnesses = []

    plays = []
    
    # run model on each board
    for b in boards:
        # default position of player
        player = b[1]

        moves = []

        # whether or not the key has been collected
        has_key = 0
        # whether or not the door has been opened
        opened_door = 0
        
        steps = 0
        
        for i in range(max_steps):
            steps += 1

            curr_item = item
            item_pos  = f.positioner(item, has_key, opened_door, b[2:])

            # if the agent has not opened the door, it cannot know where the goal is
            if not opened_door and item == 1:
                prox = (-1, -1)
            else:
                prox = f.dist(item_pos,player)

            # rotate board for diversity, flatten for compatability
            net_input = np.concatenate([b[0], [has_key], prox, player])
            move, item = net.activate(net_input)
    
            item = int(np.floor(item*(3-1e-10)))
            
            player, has_key, opened_door, score = f.make_move(curr_item, player, has_key, opened_door, move, b, dim)

            # avoid using the extra memory if not evaluating the winner
            moves.append((player,curr_item))
            
            # if goal is reached or player has fallen into pit, end run
            if score == 1 or score == -1:
                break
                
        fitnesses.append(max_steps-steps)
        plays.append((moves))
    return fitnesses, plays

def eval_genomes(genome, config):
    boards = f.generate_board(dim, rotate, board_num)
    fitnesses, plays = genome_fitness(genome, config, boards)
    fitness = f.get_fitness(fitnesses, plays)
    genome.fitness = fitness
    return genome.fitness

class parallel_evaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_genomes
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        fitnesses = []

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
            fitnesses.append(genome.fitness)

        #print('Item Mean STD:',item_std.value)
        #print('Move Mean STD:',move_std.value)
        history.append(np.mean(fitnesses))

local_dir = os.path.dirname(os.getcwd())
config_file = os.path.join(local_dir,'neat', 'config-feedforward')    # Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
            
# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
#p.add_reporter(neat.Checkpointer(5))

num_workers = 32
evaluator = parallel_evaluator(num_workers, eval_genomes)

# Run for up to 300 generations.
winner = p.run(evaluator.evaluate, 20)