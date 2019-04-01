import argparse
from process import LowPolyfier
from helpers import get_output_name
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
arguments = parser.parse_args()


def main(args):
    image_path = args.image_path
    output_path = get_output_name(image_path)
    image = Image.open(image_path)
    lowpolyfier = LowPolyfier()
    low_poly_image = lowpolyfier.lowpolyfy(image)
    low_poly_image.save(output_path, quality=100)


if __name__ == "__main__":
    main(arguments)
