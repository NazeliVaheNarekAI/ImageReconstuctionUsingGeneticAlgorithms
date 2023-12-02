from PIL import Image
import random


def blend_crossover(img1, img2, a=-1):
    """
    param img1: The first parent
    param img2: The second parent
    param a: a given opacity of (a) will be given to img1, and opacity of (1-a) to the second.
    Then they'll be added together. If a is not given, it'll be choosen randomly from (0,1)
    return: the blended child
    """
    if img1.size != img2.size:
        raise ValueError("Images must be of the same size")

    if a == -1:
        a = random.uniform(0, 1)

    blended = Image.blend(img1, img2, a)

    return blended


def row_column_slicing_crossover(img1, img2, h_prob=0.5, crossover_point=-1):
    """
    param img1: The first parent
    param img2: The second parent
    param h_prob: the probability of having a horizontal crossover, default is 0.5
    param crossover_point: the crossover point for slicing, if not provided will be choosen randomly
    return: the child
    """
    if img1.size != img2.size:
        raise ValueError("Images must be of the same size")

    if random.uniform(0, 1) < h_prob:
        # Horizontal crossover
        if crossover_point == -1:
            crossover_point = random.randint(0, img1.size[0])
        img1_cropped = img1.crop((0, 0, img1.size[0], crossover_point))
        img2_cropped = img2.crop((0, crossover_point, img2.size[0], img2.size[1]))
        result = Image.new("RGB", (img1.size[0], img1.size[1]))
        result.paste(img1_cropped, (0, 0))
        result.paste(img2_cropped, (0, crossover_point))
    else:
        # Vertical crossover
        if crossover_point == -1:
            crossover_point = random.randint(0, img1.size[1])
        img1_cropped = img1.crop((0, 0, crossover_point, img1.size[1]))
        img2_cropped = img2.crop((crossover_point, 0, img2.size[0], img2.size[1]))
        result = Image.new("RGB", (img1.size[0], img1.size[1]))
        result.paste(img1_cropped, (0, 0))
        result.paste(img2_cropped, (crossover_point, 0))

    return result


def pixel_wise_crossover(img1, img2, img1_prob=0.5):
    """
    param img1: The first parent
    param img2: The second parent
    param img1_prob: the probability of a pixel being chosen form img1, default = 0.5
    return: the child
    """
    if img1.size != img2.size:
        raise ValueError("Images must be of the same size")

    w, h = img1.size
    result = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            # Randomly choose whether to use pixel from func1 or func2
            if random.uniform(0, 1) < img1_prob:
                result.putpixel((x, y), img1.getpixel((x, y)))
            else:
                result.putpixel((x, y), img2.getpixel((x, y)))

    return result


def random_row_column_crossover(img1, img2, r_prob=0.5, img1_prob=0.5):
    """
    param img1: The first parent
    param img2: The second parent
    param r_prob: the probability of having a row crossover, default is 0.5
    param img1_prob: the probability of a row/column being chosen form img1, default = 0.5
    return: the child
    """
    if img1.size != img2.size:
        raise ValueError("Images must be of the same size")

    width, height = img1.size

    result = Image.new("RGB", (width, height))

    if random.uniform(0, 1) < r_prob:
        for i in range(height):
            if random.uniform(0, 1) < img1_prob:
                result.paste(img1.crop((0, i, width, i + 1)), (0, i))
            else:
                result.paste(img2.crop((0, i, width, i + 1)), (0, i))
    else:
        for i in range(width):
            if random.uniform(0, 1) < img1_prob:
                result.paste(img1.crop((i, 0, i + 1, height)), (i, 0))
            else:
                result.paste(img2.crop((i, 0, i + 1, height)), (i, 0))

    return result
