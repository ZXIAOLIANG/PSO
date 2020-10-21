import copy
import numpy as np
import matplotlib.pyplot as plt

def fitness_eval(x, y):
    return (4.0-2.1*pow(x,2)+pow(x,4)/3.0)*pow(x,2) + x*y + (-4.0 + 4.0*pow(y,2))*pow(y,2)

def initial_get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best):
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

def get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best, g_best, success, failure):
    fitness_sum = 0
    best_fitness = float('inf')
    new_g_best = 0
    for i in range(swarm_matrix.shape[0]):
        fitness = fitness_eval(swarm_matrix[i][0],swarm_matrix[i][1])
        pBest_fitness = fitness_eval(p_best[i][0], p_best[i][1])
        # update success and failure
        if i == g_best:
            if fitness == pBest_fitness:
                # failure
                print("failure")
                failure = failure + 1
                success = 0
            else:
                # success
                success = success + 1
                failure = 0
        # update pbest
        if fitness < pBest_fitness:
            p_best[i][0] = swarm_matrix[i][0]
            p_best[i][1] = swarm_matrix[i][1]
        # calculate sum for avg
        fitness_sum += fitness
        # update gBest
        if fitness < best_fitness:
            best_fitness = fitness
            new_g_best = i
    return fitness_sum / swarm_matrix.shape[0], best_fitness, success, failure, new_g_best

def velocity_update(velocity_matrix, swarm_matrix, p_best, c1, c2, w, max_velocity, g_best, rho):
    for i in range(velocity_matrix.shape[0]):
        if i == g_best:
            velocity_matrix[i][0] = - swarm_matrix[i][0] + swarm_matrix[g_best][0] + w * velocity_matrix[i][0] + \
                rho * (1.0 - 2.0 * np.random.uniform())
            velocity_matrix[i][1] = - swarm_matrix[i][0] + swarm_matrix[g_best][1] + w * velocity_matrix[i][1] + \
                rho * (1.0 - 2.0 * np.random.uniform())
        else:
            velocity_matrix[i][0] = w * velocity_matrix[i][0] + c1*np.random.uniform()*(p_best[i][0] - swarm_matrix[i][0]) +\
                c2*np.random.uniform()*(swarm_matrix[g_best][0] - swarm_matrix[i][0])
            velocity_matrix[i][1] = w * velocity_matrix[i][1] + c1*np.random.uniform()*(p_best[i][1] - swarm_matrix[i][1]) +\
                c2*np.random.uniform()*(swarm_matrix[g_best][1] - swarm_matrix[i][1])
        # cap the velocity
        velocity_matrix[i][0] = max(min(velocity_matrix[i][0], max_velocity), -1.0*max_velocity)
        velocity_matrix[i][1] = max(min(velocity_matrix[i][1], max_velocity), -1.0*max_velocity)

def position_update(swarm_matrix, velocity_matrix, g_best, w, rho):
    for i in range(swarm_matrix.shape[0]):
        if i == g_best:
            swarm_matrix[i][0] = swarm_matrix[g_best][0] + w * velocity_matrix[i][0] + rho*(1.0 - 2.0 * np.random.uniform())
        else:
            swarm_matrix[i][0] += velocity_matrix[i][0]
            swarm_matrix[i][1] += velocity_matrix[i][1]
        # cap the postion
        swarm_matrix[i][0] = max(min(swarm_matrix[i][0], 5), -5)
        swarm_matrix[i][1] = max(min(swarm_matrix[i][1], 5), -5)

def update_rho(rho, success, failure, epsilon_s, epsilon_f):
    if success > epsilon_s:
        rho = 2.0*rho
    elif failure > epsilon_f:
        rho = 0.5*rho
    return rho

if __name__ == "__main__":
    N = 100
    max_iteration = 200
    avg_fitness_list = []
    best_fitness_list = []
    swarm_matrix = np.random.uniform(-5,5,(N, 2))
    velocity_matrix = np.zeros((N,2))
    p_best = copy.deepcopy(swarm_matrix)
    g_best = 0
    epsilon_s = 15
    epsilon_f = 5
    success = 0
    failure = 0
    rho = 1.0
    new_gBest = 0

    c1 = 1.4944
    c2 = 1.4944
    w = 0.792
    max_velocity = 1.0
    
    swarm_matrix[0][0] = 0
    avg, best, gBest = initial_get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best)
    avg_fitness_list.append(avg)
    best_fitness_list.append(best)

    for n in range(max_iteration):
        velocity_update(velocity_matrix, swarm_matrix, p_best, c1, c2, w, max_velocity, gBest, rho)
        position_update(swarm_matrix, velocity_matrix, gBest, w, rho)
        avg, best, success, failure, new_gBest = get_avg_fitness_and_best_fitness_update_p_best(swarm_matrix, p_best, gBest, success, failure)
        old_best_fit = fitness_eval(swarm_matrix[gBest][0],swarm_matrix[gBest][1])
        new_best_fit = fitness_eval(swarm_matrix[new_gBest][0],swarm_matrix[new_gBest][1])
        # if old_best_fit == new_best_fit:
        #     # failure
        #     print("failure")
        #     failure = failure + 1
        #     success = 0
        # else:
        #     # success
        #     success = success + 1
        #     failure = 0
        if new_gBest != gBest:
            rho = 1.0
            success = 0
            failure = 0
        gBest = new_gBest
        rho = update_rho(rho, success, failure, epsilon_s, epsilon_f)
        avg_fitness_list.append(avg)
        best_fitness_list.append(best)

    print("final solution: x = {}, y = {}, z = {}".format(swarm_matrix[gBest][0], swarm_matrix[gBest][1], best_fitness_list[-1]))
    itr = [x for x in range(max_iteration+1)]
    plt.figure(1)
    plt.plot(itr, avg_fitness_list)
    plt.title("GCPSO average fitness progress")
    plt.xlabel("Iteration")
    plt.ylabel("Average fitness")
    plt.show()
    plt.figure(2)
    plt.plot(itr, best_fitness_list)
    plt.title("GCPSO best fitness progress")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.show()
        

