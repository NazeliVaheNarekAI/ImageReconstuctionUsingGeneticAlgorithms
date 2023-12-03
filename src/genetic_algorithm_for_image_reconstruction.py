import random

from PIL import Image

from RandomShapeNoise import compute_color_distribution, generate_random_shapes_image, generate_noise_from_distribution
# Importing necessary functions from the files
from crossover import crossover
from evaluationfunctions import combine_metrics
from mutations import image_mutation
from selector import selector
from tqdm import tqdm


# Defining the Genetic Algorithm for Image Reconstruction
def genetic_algorithm_image_reconstruction(target_image, initial_population, generations, mutation_prob):
    """
    A genetic algorithm for image reconstruction.

    :param target_image: The target image to reconstruct.
    :param initial_population: A list of initial population images.
    :param generations: Number of generations to run the GA.
    :param mutation_prob: Probability of mutation.
    :return: The best image from the last generation.
    """
    current_population = initial_population
    best_image = current_population[0]
    best_fitness = float('-inf')
    initial_size = len(initial_population)

    for gen in tqdm(range(generations)):
        # Fitness Evaluation
        fitness_scores = []

        for individual in current_population:
            fitness = combine_metrics({'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}, individual,
                                      target_image)
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                print("New Best Fitness", fitness)
                best_fitness = fitness
                best_image = individual

        paired_fitness = list(zip(current_population, fitness_scores))

        # Sort the pairs by fitness score
        sorted_by_fitness = sorted(paired_fitness, key=lambda x: x[1], reverse=True)
        sorted_by_fitness = sorted_by_fitness[:initial_size]
        current_population = [t[0] for t in sorted_by_fitness]

        # Selection
        selected_pairs = selector(sorted_by_fitness, method='best')

        if gen % 5000 == 0:
            best_image.save('images/mona_lisa_result_' + str(gen) + '.jpg')
        # Crossover and Mutation
        next_generation = []
        if len(selected_pairs) % 2 == 1:
            selected_pairs.append(selected_pairs[-1])

        results = []
        for pair in selected_pairs:
            results.append(process_pair(pair, mutation_prob))
        for child in results:
            # child.show()
            current_population.append(child)
    return best_image


def process_pair(pair, mutation_prob):
    img1, img2 = pair[0][0], pair[1][0]
    # Crossover
    child = crossover(img1, img2, {'blend': 0.6, 'row_column_slicing': 0.35, 'pixel_wise': 0.05,
                                   'random_row_column': 0})
    # Mutation
    if random.random() < mutation_prob:
        child = image_mutation(child, pixel_mutation_prob=0.5, shape_mutation_prob=0,
                               pixel_altering_prob=0.5, max_pixel_range=255, number_of_shapes=50)
    return child


# Dummy code to simulate initial population and target image (to be replaced with actual images)
def dummy_initial_population(image, size, num_individuals):
    unique_colors, frequency = compute_color_distribution(image)
    # return [generate_noise_from_distribution(unique_colors, frequency, size) for _ in range(num_individuals)]

    # use shapes
    return [generate_random_shapes_image(unique_colors, size, 100) for _ in range(num_individuals)]


# Simulating the genetic algorithm (this is a dummy simulation, actual implementation will require real images)
if __name__ == '__main__':
    target_image = Image.open('images/download.jpeg')
    dummy_population = dummy_initial_population(target_image, target_image.size, 500)

    result = genetic_algorithm_image_reconstruction(target_image, dummy_population, generations=400,
                                                    mutation_prob=0.05)
    print(result)
    result.save('images/download_result.jpeg')
