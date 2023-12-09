from PIL import Image, ImageDraw
import numpy as np
import random


def compute_color_distribution(image):
    data = np.array(image)

    unique, counts = np.unique(data.reshape(-1, 3), axis=0, return_counts=True)
    frequency = counts / counts.sum()

    return unique, frequency


def generate_noise_from_distribution(unique_colors, frequency, image_size):
    # Choose indices based on the frequency distribution
    indices = np.random.choice(len(unique_colors), size=image_size[0] * image_size[1], p=frequency)

    # Use advanced indexing to construct the noise image data
    noise_image_data = unique_colors[indices]

    # Reshape the noise image data and convert it to an image
    noise_image = noise_image_data.reshape(image_size[1], image_size[0], 3)
    return Image.fromarray(np.uint8(noise_image))


def process_image(image_path):
    image = Image.open(image_path)

    unique_colors, frequency = compute_color_distribution(image)

    noise_image = generate_noise_from_distribution(unique_colors, frequency, image.size)

    return noise_image


def generate_random_shapes_image(unique_colors, image_size, number_of_shapes):
    bg_color = tuple(random.choice(unique_colors))

    new_image = Image.new('RGB', image_size, color=bg_color)
    draw = ImageDraw.Draw(new_image)

    for _ in range(number_of_shapes):
        color = tuple(random.choice(unique_colors))

        while color == bg_color:
            color = tuple(random.choice(unique_colors))

        x1, y1 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
        x2, y2 = random.randint(x1, image_size[0]), random.randint(y1, image_size[1])

        shape_types = [
            'rectangle', 'ellipse', 'circle', 'triangle',
            'polygon', 'line', 'arc', 'chord'
        ]
        # shape_types = [
        #     'triangle',
        # ]
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

    return new_image
