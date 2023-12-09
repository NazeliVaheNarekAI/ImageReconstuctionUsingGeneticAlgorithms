import numpy as np
import random
from PIL import Image, ImageDraw


def image_mutation(image, pixel_mutation_prob, shape_mutation_prob, pixel_altering_prob, max_pixel_range,
                   number_of_shapes=1, unique_colors=None, frequency=None):
    """
    Apply pixel-wise mutation and/or random shape mutation to an image based on given probabilities.

    :param frequency: The frequency of each color in the image.
    :param unique_colors: The unique colors in the image.
    :param max_pixel_range: Maximum value to add to a pixel for the pixel-wise mutation.
    :param pixel_altering_prob: Probability to change a pixel (between 0 and 1)
    :param image: PIL Image object.
    :param pixel_mutation_prob: Probability to apply pixel-wise mutation.
    :param shape_mutation_prob: Probability to apply random shape mutation.
    :param number_of_shapes: Number of shapes to add for the shape mutation.
    :return: Mutated PIL Image object.
    """
    # Apply pixelwise mutation with given probability
    if random.random() < pixel_mutation_prob:
        image = pixel_wise_mutation(image, pixel_altering_prob, max_pixel_range)

    # Apply random shape mutation with given probability
    if random.random() < shape_mutation_prob:
        image = add_random_shape_mutation(image, number_of_shapes, unique_colors, frequency)

    return image


def pixel_wise_mutation(image, probability, max_value):
    """
    :param image: A PIL Image object representing the input image.
    :param probability: The probability of a pixel being mutated. Must be a float between 0 and 1.
    :param max_value: The maximum value that a pixel can take. Must be an integer between 0 and 255.
    :return: A mutated PIL Image object.

    This method performs pixel-wise mutation on the input image. It randomly alters the pixel values based on the given probability and maximum value.

    The method first converts the input image to a numpy array. It then generates a matrix of random values and a boolean mask with the same shape as the image. The random values are integers
    * between 0 and the maximum value. The boolean mask is created based on the given probability, where a True value indicates a pixel to be mutated.

    The method uses numpy's element-wise addition and conditional expression to safely add random values to the image only where the mask is True. It ensures that the resulting values are
    * within the correct range by clipping them between 0 and 255.

    Finally, the method converts the mutated numpy array back to a PIL Image object and returns it.

    Example usage:
    image = Image.open('input.jpg')
    mutated_image = pixel_wise_mutation(image, 0.5, 255)
    mutated_image.save('output.jpg')
    """
    # Convert image to numpy array
    img_array = np.array(image)

    # Generate a matrix of random values and a boolean mask
    random_values = np.random.randint(0, max_value, img_array.shape)
    probability_mask = np.random.random(img_array.shape[:2]) < probability

    # Safely add random values where the mask is True
    # Ensure the values are within the correct range
    img_array = np.clip(
        img_array + np.where(probability_mask[..., None], np.random.randint(0, max_value, img_array.shape), 0), 0, 255)

    # Convert back to PIL Image
    altered_image = Image.fromarray(img_array.astype('uint8'))

    return altered_image


def add_random_shape_mutation(image, number_of_shapes=1, unique_colors=None, frequency=None):
    """
    Add Random Shape Mutation

    Add random shapes to an image.

    :param image: The original image to which shapes will be added.
    :type image: PIL.Image.Image

    :param number_of_shapes: The number of shapes to add. Default is 1.
    :type number_of_shapes: int

    :param unique_colors: List of unique colors to be used for the shapes. Default is None.
    :type unique_colors: list, optional

    :param frequency: List of frequencies corresponding to the unique colors. Default is None.
    :type frequency: list, optional

    :return: The image with the added random shapes.
    :rtype: PIL.Image.Image

    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    image_size = image.size

    # Get image dimensions
    for _ in range(number_of_shapes):
        # color = tuple([random.randint(0, 255) for _ in range(3)])
        color = random.choices(unique_colors, weights=frequency)
        color = tuple(color[0])

        image_size = image.size

        x1, y1 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        x2, y2 = random.randint(x1, image_size[0]), random.randint(y1, image_size[1])

        shape_types = [
            'rectangle', 'ellipse', 'circle', 'triangle',
            'polygon', 'line', 'arc', 'chord'
        ]
        shape_type = random.choice(shape_types)

        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == 'circle':
            radius = min(x2 - x1, y2 - y1) // 2
            center = (x1 + radius, y1 + radius)
            draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=color)
        elif shape_type == 'triangle':
            draw.polygon([x1, y2, (x1 + x2) // 2, y1, x2, y2], fill=color)
        elif shape_type == 'polygon':
            num_points = random.randint(3, 10)  # Polygons with 3 to 10 sides
            points = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(num_points)]
            draw.polygon(points, fill=color)
        elif shape_type == 'line':
            draw.line([x1, y1, x2, y2], fill=color, width=3)
        elif shape_type == 'arc':
            draw.arc([x1, y1, x2, y2], start=0, end=180, fill=color)
        elif shape_type == 'chord':
            draw.chord([x1, y1, x2, y2], start=0, end=180, fill=color)
    return image_copy
