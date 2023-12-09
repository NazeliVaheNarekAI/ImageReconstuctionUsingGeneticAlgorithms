import random

from PIL import Image

from population import compute_color_distribution, generate_random_shapes_image, generate_noise_from_distribution
# Importing necessary functions from the files
from crossover import crossover
from evaluationfunctions import combine_metrics
from mutations import image_mutation
import selector as selector
from tqdm import tqdm
import methods as methods


def gen_alg(target_image):
    """
    :param target_image: The image that the genetic algorithm will attempt to reconstruct.
    :return: The genetically reconstructed image that closely resembles the target image.
    """
    return genetic_algorithm_image_reconstruction(target_image, generations=250,
                                                  mutation_prob=0.1, initial_population_size=50)


# Defining the Genetic Algorithm for Image Reconstruction
def genetic_algorithm_image_reconstruction(target_image, generations, mutation_prob, initial_population_size):
    """
    A genetic algorithm for image reconstruction.

    :param target_image: The target image to reconstruct.
    :param initial_population: A list of initial population images.
    :param generations: Number of generations to run the GA.
    :param mutation_prob: Probability of mutation.
    :return: The best image from the last generation.
    """

    unique_colors, frequency = compute_color_distribution(target_image)

    initial_population = create_population(unique_colors, frequency, target_image.size, initial_population_size)
    current_population = initial_population
    best_image = current_population[0]
    best_fitness = float('-inf')
    initial_size = len(initial_population)

    for gen in tqdm(range(generations)):
        # Fitness Evaluation
        fitness_scores = []

        for individual in current_population:
            fitness = calculate_fitness(individual, target_image)
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                print("New Best Fitness", fitness)
                best_fitness = fitness
                best_image = individual

        # Selection
        # selected_pairs = selector(sorted_by_fitness, method='best')
        selected_pairs = selector.get_parents(current_population, fitness_scores)

        if gen % 50 == 0:
            best_image.save('images/mona_lisa_result_' + str(gen) + '.jpg')

        results = []
        for pair in selected_pairs:
            results.append(process_pair(pair, mutation_prob, unique_colors, frequency))
        for child in results:
            # child.show()
            current_population.append(child)
            fitness_scores.append(calculate_fitness(child, target_image))

        # Elitism
        sorted_by_fitness = sorted(zip(current_population, fitness_scores), key=lambda x: x[1], reverse=True)
        current_population = [x[0] for x in sorted_by_fitness[:initial_size]]

    return best_image


def calculate_fitness(image, target_image):
    """
    Calculate the fitness of an image compared to the target image.

    :param image: The image to calculate the fitness of.
    :param target_image: The target image.
    :return: The fitness score.
    """
    return combine_metrics({'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}, image, target_image)


def process_pair(pair, mutation_prob, unique_colors, frequency):
    """
    Process a pair of images.

    :param pair: The pair of images.
    :param mutation_prob: The probability of mutation.
    :param unique_colors: The number of unique colors in the images.
    :param frequency: The frequency of the images.
    :return: The processed child image.
    """
    img1, img2 = pair[0], pair[1]
    # Crossover
    child = crossover(img1, img2, {'blend': 0, 'row_column_slicing': 1, 'pixel_wise': 0,
                                   'random_row_column': 0})
    # Mutation
    if random.random() < mutation_prob:
        child = image_mutation(child, pixel_mutation_prob=0.3, shape_mutation_prob=0.7,
                               pixel_altering_prob=0.5, max_pixel_range=125, number_of_shapes=3,
                               unique_colors=unique_colors, frequency=frequency)
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


if __name__ == '__main__':
    use_chunking_mode = True   # set this to True to use the Chunking modem, False to use the normal mode

    if not use_chunking_mode:
        target_image = Image.open('images/LittlePrince.jpg')
        initial_size = target_image.size
        target_image = target_image.resize((200, target_image.size[1] * 200 // target_image.size[0]))
        result = genetic_algorithm_image_reconstruction(target_image, generations=16000,
                                                        mutation_prob=0.2, initial_population_size=50)
        result = result.resize(initial_size)
        result.save('images/result.jpeg')

    else:
        target_image = Image.open('images/Dior.jpg')
        base_size = 600
        resized_target = target_image.resize((base_size, target_image.size[1] * base_size // target_image.size[0]))
        print(resized_target.size)
        result = methods.chunking_with_padding(resized_target, 25, 25, gen_alg, blend_width=10)
        result = result.resize(target_image.size)
        result.save('images/result.jpeg')
