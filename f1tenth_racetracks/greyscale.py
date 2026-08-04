from pathlib import Path

from PIL import Image

from config import load_racetrack_config

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

def main():
    module = Path(__file__).resolve().parent
    config = load_racetrack_config().greyscale
    convert_to_greyscale(
        module / config.input_path, module / config.output_path
    )


if __name__ == '__main__':
    main()
