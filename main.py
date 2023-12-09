from PIL import Image

from src import methods
from src.genetic_algorithm_for_image_reconstruction import genetic_algorithm_image_reconstruction, gen_alg_for_chunking

if __name__ == '__main__':
    use_chunking_mode = False  # set this to True to use the Chunking mode, False to use the normal mode

    fitness = {'psnr': False, 'ssim': False, 'delta_e': False, 'mse': True}
    mutation = {'pixel_mutation_prob': 0.3, 'shape_mutation_prob': 0.7, 'pixel_altering_prob': 0.5,
                'max_pixel_range': 125, 'number_of_shapes': 3}
    if not use_chunking_mode:
        target_image = Image.open('images/LittlePrince.jpg')
        initial_size = target_image.size
        target_image = target_image.resize((200, target_image.size[1] * 200 // target_image.size[0]))
        result = genetic_algorithm_image_reconstruction(target_image, generations=16000,
                                                        mutation_prob=0.2, initial_population_size=50,
                                                        fitness_options=fitness, mutation_options=mutation)
        result = result.resize(initial_size)
        result.save('images/result.jpeg')

    else:
        target_image = Image.open('images/Dior.jpg')
        base_size = 600
        resized_target = target_image.resize((base_size, target_image.size[1] * base_size // target_image.size[0]))
        print(resized_target.size)
        result = methods.chunking_with_padding(resized_target, 25, 25, gen_alg_for_chunking, blend_width=10)
        result = result.resize(target_image.size)
        result.save('images/result.jpeg')
