def selector(populations, fit_func, method='best'):
    fitness_scores = [(population, fit_func(population)) for population in populations]

    sorted_by_fitness = sorted(fitness_scores, key=lambda x: x[1], reverse=(method == 'best'))

    paired_populations = []
    for i in range(0, len(sorted_by_fitness), 2):
        paired_populations.append(tuple(sorted_by_fitness[i:i + 2]))

    return paired_populations