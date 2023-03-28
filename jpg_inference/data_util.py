from PIL import Image
import math
import time

def padded_resize(image, target_width, target_height, fill_color=(0, 0, 0)):
    """
    Resizes an image to fit within the given target dimensions while maintaining aspect ratio, and adds padding
    to fill any remaining empty space.
    
    Parameters:
    image (PIL.Image): The image to resize.
    target_width (int): The desired width of the target container or display area.
    target_height (int): The desired height of the target container or display area.
    fill_color (tuple): The RGB color tuple to use for the padding (default is white).
    
    Returns:
    PIL.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    
    # Determine the scaling factor for each dimension
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    
    # Use the smaller scaling factor to ensure that the entire image fits within the target dimensions
    scale_factor = min(width_ratio, height_ratio)
    
    # Resize the image while maintaining aspect ratio
    new_width = math.floor(original_width * scale_factor)
    new_height = math.floor(original_height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    
    # Add padding to fill any remaining empty space
    padding_width = target_width - new_width
    padding_height = target_height - new_height
    
    padding_left = math.floor(padding_width / 2)
    padding_top = math.floor(padding_height / 2)
    
    padded_image = Image.new(image.mode, (target_width, target_height), fill_color)
    padded_image.paste(resized_image, (padding_left, padding_top))
    
    return padded_image


