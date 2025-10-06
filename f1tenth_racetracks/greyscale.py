from PIL import Image

def convert_to_greyscale(input_path, output_path):
    """
    Converts an image to greyscale and saves the result.
    
    :param input_path: Path to the input image
    :param output_path: Path to save the greyscale image
    """
    # Open the image
    image = Image.open(input_path)
    
    # Convert the image to greyscale
    greyscale_image = image.convert("L")
    
    # Save the greyscale image
    greyscale_image.save(output_path)
    print(f"Greyscale image saved to {output_path}")

# Example usage
input_image_path = "Austin/Untitled Diagram.drawio.png"
output_image_path = "Austin/Austin_map_block.png"
convert_to_greyscale(input_image_path, output_image_path)

