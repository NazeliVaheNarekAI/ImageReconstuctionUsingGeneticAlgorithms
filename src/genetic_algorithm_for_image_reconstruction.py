import random

from tqdm import tqdm

import src.selector as selector
# Importing necessary functions from the files
from src.crossover import crossover
from src.evaluationfunctions import combine_metrics
from src.mutations import image_mutation
from src.population import compute_color_distribution, generate_random_shapes_image


# Defining the Genetic Algorithm for Image Reconstruction
def genetic_algorithm_image_reconstruction(target, generations, mutation_prob, initial_population_size,
                                           fitness_options=None,
                                           mutation_options=None):
    """
    Reconstructs an image using genetic algorithm.

    :param target: The target image to reconstruct.
    :type target: PIL.Image.Image
    :param generations: The number of generations to evolve.
    :type generations: int
    :param mutation_prob: The probability of mutation.
    :type mutation_prob: float
    :param initial_population_size: The initial size of the population.
    :type initial_population_size: int
    :param fitness_options: Options for fitness evaluation.
    :type fitness_options: dict, optional
    :param mutation_options: Options for mutation.
    :type mutation_options: dict, optional
    :return: The reconstructed image.
    :rtype: PIL.Image.Image
    """

    if mutation_options is None:
        mutation_options = {'pixel_mutation_prob': 0.3, 'shape_mutation_prob': 0.7,
                            'pixel_altering_prob': 0.5, 'max_pixel_range': 125,
                            'number_of_shapes': 3}
    if fitness_options is None:
        fitness_options = {'psnr': False, 'ssim': False, 'delta_e': False,
                           'mse': True}

    unique_colors, frequency = compute_color_distribution(target)

    initial_population = create_population(unique_colors, frequency, target.size, initial_population_size)
    current_population = initial_population
    best_image = current_population[0]
    best_fitness = float('-inf')
    initial_size = len(initial_population)

    for gen in tqdm(range(generations)):
        # Fitness Evaluation
        fitness_scores = []

        for individual in current_population:
            fitness = calculate_fitness(individual, target, fitness_options)
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                print("New Best Fitness", fitness)
                best_fitness = fitness
                best_image = individual

        # Selection
        # selected_pairs = selector(sorted_by_fitness, method='best')
        selected_pairs = selector.get_parents(current_population, fitness_scores)

        if gen % 50 == 0:
            best_image.save('images/temp/result_' + str(gen) + '.jpg')

        results = []
        for pair in selected_pairs:
            results.append(process_pair(pair, mutation_prob, unique_colors, frequency, **mutation_options))
        for child in results:
            # child.show()
            current_population.append(child)
            fitness_scores.append(calculate_fitness(child, target, fitness_options))

        # Elitism
        sorted_by_fitness = sorted(zip(current_population, fitness_scores), key=lambda x: x[1], reverse=True)
        current_population = [x[0] for x in sorted_by_fitness[:initial_size]]

    return best_image


def calculate_fitness(image, target, options=None):
    """
    Calculate the fitness of an image compared to a target image.

    :param image: The image that will be evaluated.
    :param target: The target image to compare against.
    :param options: A dictionary containing the options for calculating fitness. Default is {'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}.
    :return: The calculated fitness value based on the given options.
    """
    if options is None:
        options = {'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}
    return combine_metrics(options, image, target)


def process_pair(pair, mutation_prob, unique_colors, frequency, pixel_mutation_prob=0.3, shape_mutation_prob=0.7,
                 pixel_altering_prob=0.5, max_pixel_range=125, number_of_shapes=3):
    """
    Process a pair of images.

    :param pair: A tuple containing two images to be processed.
    :param mutation_prob: The probability of applying mutation to the child image.
    :param unique_colors: A list of unique colors used in the images.
    :param frequency: A dictionary containing the frequency of each color in the images.
    :param pixel_mutation_prob: The probability of applying pixel-level mutation.
    :param shape_mutation_prob: The probability of applying shape-level mutation.
    :param pixel_altering_prob: The probability of altering each individual pixel value.
    :param max_pixel_range: The maximum range of pixel values for altering pixel values.
    :param number_of_shapes: The number of shapes to be added in shape-level mutation.
    :return: The processed child image.
    """
    img1, img2 = pair[0], pair[1]
    # Crossover
    child = crossover(img1, img2, {'blend': 0, 'row_column_slicing': 1, 'pixel_wise': 0,
                                   'random_row_column': 0})
    # Mutation
    if random.random() < mutation_prob:
        child = image_mutation(child, pixel_mutation_prob, shape_mutation_prob,
                               pixel_altering_prob, max_pixel_range, number_of_shapes,
                               unique_colors, frequency)
    return child


# Dummy code to simulate initial population and target image (to be replaced with actual images)
def create_population(unique_colors, frequency, size, num_individuals):
    """
    Create a population of individuals with random shapes images.

    :param unique_colors: A list of unique colors.
    :param frequency: A list of frequencies of each color.
    :param size: The size of the images.
    :param num_individuals: The number of individuals in the population.
    :return: A list of random shapes images.
    """
    # return [generate_noise_from_distribution(unique_colors, frequency, size) for _ in range(num_individuals)]

    # use shapes
    return [generate_random_shapes_image(unique_colors, size, 5) for _ in range(num_individuals)]
    # return [generate_noise_from_distribution(unique_colors, frequency, size) for _ in range(num_individuals)]


def gen_alg_for_chunking(target, generations=250, mutation_prob=0.1, initial_population_size=50):
    """
    :param target: The image that needs to be reconstructed.
    :param generations: The number of generations for the genetic algorithm. Default value is 250.
    :param mutation_prob: The probability of mutation for each individual in the population. Default value is 0.1.
    :param initial_population_size: The size of the initial population for the genetic algorithm. Default value is 50.
    :return: The reconstructed image obtained using the genetic algorithm.

    """
    fit = {'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}
    mut = {'pixel_mutation_prob': 0.3, 'shape_mutation_prob': 0.7, 'pixel_altering_prob': 0.5,
           'max_pixel_range': 125, 'number_of_shapes': 3}

    return genetic_algorithm_image_reconstruction(target, generations, mutation_prob, initial_population_size,
                                                  fit, mut)


