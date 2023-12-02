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

def selector(populations, fitness_values, method='best'):
    # Pair each population with its fitness score
    paired_fitness = list(zip(populations, fitness_values))

    # Sort the pairs by fitness score
    sorted_by_fitness = sorted(paired_fitness, key=lambda x: x[1], reverse=(method == 'best'))

    # Pair the populations for selection
    paired_populations = []
    for i in range(0, len(sorted_by_fitness), 2):
        paired_populations.append(tuple(sorted_by_fitness[i:i + 2]))

    return paired_populations
