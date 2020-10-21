import copy
import numpy as np
import matplotlib.pyplot as plt

def fitness_eval(x, y):
    return (4.0-2.1*pow(x,2)+pow(x,4)/3.0)*pow(x,2) + x*y + (-4.0 + 4.0*pow(y,2))*pow(y,2)

def get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best):
    fitness_sum = 0
    best_fitness = float('inf')
    gBest = 0
    for i in range(swarm_matrix.shape[0]):
        fitness = fitness_eval(swarm_matrix[i][0],swarm_matrix[i][1])
        # update pbest
        pBest_fitness = fitness_eval(p_best[i][0], p_best[i][1])
        if fitness < pBest_fitness:
            p_best[i][0] = swarm_matrix[i][0]
            p_best[i][1] = swarm_matrix[i][1]
        # calculate sum for avg
        fitness_sum += fitness
        # update gBest
        if fitness < best_fitness:
            best_fitness = fitness
            gBest = i
    return fitness_sum / swarm_matrix.shape[0], best_fitness, gBest

def velocity_update(velocity_matrix, swarm_matrix, p_best, c1, c2, max_velocity, g_best):
    for i in range(velocity_matrix.shape[0]):
        velocity_matrix[i][0] = velocity_matrix[i][0] + c1*np.random.uniform()*(p_best[i][0] - swarm_matrix[i][0]) +\
            c2*np.random.uniform()*(swarm_matrix[g_best][0] - swarm_matrix[i][0])
        velocity_matrix[i][1] = velocity_matrix[i][1] + c1*np.random.uniform()*(p_best[i][1] - swarm_matrix[i][1]) +\
            c2*np.random.uniform()*(swarm_matrix[g_best][1] - swarm_matrix[i][1])
        # cap the velocity
        velocity_matrix[i][0] = max(min(velocity_matrix[i][0], max_velocity), -1.0*max_velocity)
        velocity_matrix[i][1] = max(min(velocity_matrix[i][1], max_velocity), -1.0*max_velocity)

def position_update(swarm_matrix, velocity_matrix):
    for i in range(swarm_matrix.shape[0]):
        swarm_matrix[i][0] += velocity_matrix[i][0]
        swarm_matrix[i][1] += velocity_matrix[i][1]
        # cap the postion
        swarm_matrix[i][0] = max(min(swarm_matrix[i][0], 5), -5)
        swarm_matrix[i][1] = max(min(swarm_matrix[i][1], 5), -5)


if __name__ == "__main__":
    N = 100
    max_iteration = 200
    avg_fitness_list = []
    best_fitness_list = []
    swarm_matrix = np.random.uniform(-5,5,(N, 2))
    velocity_matrix = np.zeros((N,2))
    p_best = copy.deepcopy(swarm_matrix)
    g_best = 0
    c1 = 1.4944
    c2 = 1.4944
    max_velocity = 1.0
    
    swarm_matrix[0][0] = 0
    avg, best, gBest = get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best)
    avg_fitness_list.append(avg)
    best_fitness_list.append(best)

    for n in range(max_iteration):
        velocity_update(velocity_matrix, swarm_matrix, p_best, c1, c2, max_velocity, gBest)
        position_update(swarm_matrix, velocity_matrix)
        avg, best, gBest  = get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best)
        avg_fitness_list.append(avg)
        best_fitness_list.append(best)

    print("final solution: x = {}, y = {}, z = {}".format(swarm_matrix[gBest][0], swarm_matrix[gBest][1], best_fitness_list[-1]))
    itr = [x for x in range(max_iteration+1)]
    plt.figure(1)
    plt.plot(itr, avg_fitness_list)
    plt.title("Simple PSO average fitness progress")
    plt.xlabel("Iteration")
    plt.ylabel("Average fitness")
    plt.show()
    plt.figure(2)
    plt.plot(itr, best_fitness_list)
    plt.title("Simple PSO best fitness progress")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.show()
        

