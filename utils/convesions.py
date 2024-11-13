def convert_pixels_to_meters(pixels, reference_height_in_meters, reference_height_in_pixels):
    return (pixels * reference_height_in_meters) / reference_height_in_pixels


def convert_meters_to_pixels(meters, reference_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / reference_height_in_meters