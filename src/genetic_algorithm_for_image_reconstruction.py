import random

from PIL import Image

from RandomShapeNoise import compute_color_distribution, generate_random_shapes_image
# Importing necessary functions from the files
from crossover import blend_crossover
from evaluationfunctions import combine_metrics
from mutations import image_mutation
from selector import selector


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
    best_image = None
    best_fitness = float('inf')

    for generation in range(generations):
        # Fitness Evaluation
        fitness_scores = []
        for individual in current_population:
            fitness = combine_metrics({'mse': True}, individual, target_image)
            fitness_scores.append((individual, fitness))
            if fitness < best_fitness:
                best_fitness = fitness
                best_image = individual

        # Selection
        selected_pairs = selector(fitness_scores, lambda x: x[1], method='best')

        print(selected_pairs[0][0])
        # Crossover and Mutation
        next_generation = []
        for pair in selected_pairs:
            img1, img2 = pair[0], pair[0]
            # Crossover
            child = blend_crossover(img1, img2)
            # Mutation
            if random.random() < mutation_prob:
                child = image_mutation(child, pixel_mutation_prob=0.1, shape_mutation_prob=0.1,
                                       pixel_altering_prob=0.1, max_pixel_range=255, number_of_shapes=1)
            next_generation.append(child)

        current_population = next_generation

    return best_image


# Dummy code to simulate initial population and target image (to be replaced with actual images)
def dummy_initial_population(image, size, num_individuals):
    unique_colors, frequency = compute_color_distribution(image)
    # return [generate_noise_from_distribution(unique_colors, frequency, size) for _ in range(num_individuals)]

    #use shapes
    return [generate_random_shapes_image(unique_colors, size, 10) for _ in range(num_individuals)]


# Simulating the genetic algorithm (this is a dummy simulation, actual implementation will require real images)
if __name__ == '__main__':
    target_image = Image.open('images/mona_lisa.jpg')
    dummy_population = dummy_initial_population(target_image, target_image.size, 100)

    # save the first image in the population
    dummy_population[0].save('images/mona_lisa_0.jpg')
    genetic_algorithm_image_reconstruction(target_image, dummy_population, generations=10, mutation_prob=0.2)