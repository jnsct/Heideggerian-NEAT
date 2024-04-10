import os
import neat
import pickle

import functions as f

size        = 7    # size of board
board_num   = 50   # how many boards to send train each genome on, will be x4 if rotation is on
max_steps   = 15   # how many steps the agent is allowed to take before ending the simulation
generations = 1000 # how many generations to train
save        = True # optionally save the genome to pickle file


local_dir = os.path.dirname(os.getcwd())                           # define working directory
config_file = os.path.join(local_dir,'neat', 'config-feedforward') # load the config file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
            
# create population
p = neat.Population(config)

# add an stdout reporter to show progress in terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# define number of parallel workers, should be set between 1 and number of CPUs in system
num_workers = 32
evaluator = f.parallel_evaluator(num_workers, f.eval_genomes, size, board_num, max_steps)

# run for specified number of generations
winner = p.run(evaluator.evaluate, generations)

history = evaluator.get_history()

# save the best genome to a file
if save:
    with open('best_genome.pkl', 'wb') as file:
        pickle.dump((winner, history), file)
