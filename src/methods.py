from PIL import Image

def chunking_with_padding(img, target, rows, cols, ga_function, blend_width=10, padding=(10, 10, 10, 10)):
    """
    The function divides the given image into chunks with padding,
    applies a GA to them individually, and blends the overlapping regions.

    :param img: The initial image
    :param target: The target image
    :param rows: How many chunks we want to have horizontally
    :param cols: How many chunks we want to have vertically
    :param ga_function: The genetic algorithm function
    :param blend_width: The width of the blending region
    :param padding: Padding from the sides (left, right, upper, lower) when processing chunks
    :return: The image after chunking with GA and blending
    """
    if img.size != target.size:
        raise ValueError("Images must be of the same size")

    width, height = img.size

    if width % cols != 0 or height % rows != 0:
        raise ValueError("Number of rows and columns must evenly divide the image dimensions")

    chunk_width = width // cols
    chunk_height = height // rows

    res_chunks = []

    for i in range(rows):
        for j in range(cols):
            left = j * chunk_width
            upper = i * chunk_height
            right = left + chunk_width
            lower = upper + chunk_height

            # Add padding to the coordinates
            padded_left = max(left - padding[0], 0)
            padded_upper = max(upper - padding[2], 0)
            padded_right = min(right + padding[1], width)
            padded_lower = min(lower + padding[3], height)

            chunk = img.crop((padded_left, padded_upper, padded_right, padded_lower))
            target_chunk = target.crop((padded_left, padded_upper, padded_right, padded_lower))

            # Apply the genetic algorithm function to the padded chunk
            res_chunk = ga_function(chunk, target_chunk)

            res_chunks.append(res_chunk)

    result = Image.new("RGB", (width, height))

    for i in range(rows):
        for j in range(cols):
            left = j * chunk_width
            upper = i * chunk_height

            # Add padding to the coordinates
            padded_left = max(left - padding[0], 0)
            padded_upper = max(upper - padding[2], 0)
            padded_right = min(left + chunk_width + padding[1], width)
            padded_lower = min(upper + chunk_height + padding[3], height)

            # Paste the processed chunk into the result image
            result.paste(res_chunks[i * cols + j], (padded_left, padded_upper))

            # Blend the right edge with the next chunk
            if j < cols - 1:
                next_chunk_left = (j + 1) * chunk_width
                blend_width_right = min(blend_width, next_chunk_left - left)
                blend_region = res_chunks[i * cols + j].crop((chunk_width - blend_width_right, 0, chunk_width, padded_lower - padded_upper))
                next_chunk_region = res_chunks[i * cols + (j + 1)].crop((0, 0, blend_width_right, padded_lower - padded_upper))
                blended_edge = Image.blend(blend_region, next_chunk_region, 0.5)
                result.paste(blended_edge, (padded_left + chunk_width - blend_width_right, padded_upper))

            # Blend the bottom edge with the next row
            if i < rows - 1:
                next_row_upper = (i + 1) * chunk_height
                blend_height_bottom = min(blend_width, next_row_upper - upper)
                blend_region = res_chunks[i * cols + j].crop((0, chunk_height - blend_height_bottom, chunk_width, chunk_height))
                next_row_region = res_chunks[(i + 1) * cols + j].crop((0, 0, chunk_width, blend_height_bottom))
                blended_edge = Image.blend(blend_region, next_row_region, 0.5)
                result.paste(blended_edge, (padded_left, padded_upper + chunk_height - blend_height_bottom))

    return result