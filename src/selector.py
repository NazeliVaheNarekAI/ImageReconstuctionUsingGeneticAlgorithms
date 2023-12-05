# def selector(populations, fit_func, method='best'):
#     fitness_scores = [(population, fit_func(population)) for population in populations]
#
#     sorted_by_fitness = sorted(fitness_scores, key=lambda x: x[1], reverse=(method == 'best'))
#
#     paired_populations = []
#     for i in range(0, len(sorted_by_fitness), 2):
#         paired_populations.append(tuple(sorted_by_fitness[i:i + 2]))
#
#     return paired_populations
import random
import numpy as np


# def selector(populations, fitness_values, method='best', max_population_size=50):
#     # Pair each population with its fitness score
#     paired_fitness = list(zip(populations, fitness_values))
#
#     # Sort the pairs by fitness score
#     sorted_by_fitness = sorted(paired_fitness, key=lambda x: x[1], reverse=(method == 'best'))
#
#     # Pair the populations for selection
#     paired_populations = []
#     for i in range(0, len(sorted_by_fitness), 2):
#         paired_populations.append(tuple(sorted_by_fitness[i:i + 2]))
#
#     return paired_populations

def selector(sorted_by_fitness, method='best', max_population_size=50):
    # Pair each population with its fitness score
    # paired_fitness = list(zip(populations, fitness_values))
    #
    # # Sort the pairs by fitness score
    # sorted_by_fitness = sorted(paired_fitness, key=lambda x: x[1], reverse=(method == 'best'))

    copy = sorted_by_fitness.copy()
    random.shuffle(copy)
    # Pair the populations for selection
    paired_populations = []
    for i in range(0, len(copy), 2):
        paired_populations.append(tuple(copy[i:i + 2]))

    return paired_populations


def get_parents(local_population, local_fitnesses):
    """Connect parents in pairs based on fitnesses as weights using softmax."""
    fitness_sum = sum(np.exp(local_fitnesses))
    fitness_normalized = np.exp(local_fitnesses) / fitness_sum
    local_parents_list = []
    for _ in range(0, len(local_population)):
        parents = random.choices(local_population, weights=fitness_normalized, k=2)
        local_parents_list.append(parents)
    return local_parents_list
