# LowPolyPy
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/maschere/6c789d70bbdaed2d89e1742f9d50a508/lowpolypy.ipynb)

Generate Low-Poly Versions of Images:

![Samples](samples.jpg)

## Work In Progress

This repo is being overhauled and new features are being added.

The Colab notebook and setup instructions may not work at the moment.

## Run it on Colab
[Run this notebook in Google Colab](https://colab.research.google.com/gist/maschere/6c789d70bbdaed2d89e1742f9d50a508/lowpolypy.ipynb) to get started quickly.

Thanks to [@maschere](https://gist.github.com/maschere/6c789d70bbdaed2d89e1742f9d50a508) for putting this notebook together!

## Run Locally

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

I'm currently working on wrapping these scripts in a Django project and getting it hosted on Heroku.

### Prerequisites

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/tasercake/lowpolypy
cd lowpolypy
```

### Installing

> Lowpolypy can't be installed via `pip` (yet), but the setup is simple nonetheless.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Try It Out

From the root of the project directory, run

```bash
python -m lowpolypy run './images/giraffe.jpg'
```

This should create a new image in the `images` directory.

## Built With

* [OpenCV](https://opencv.org/releases/) - Image manipulation library
* [PIL](https://pillow.readthedocs.io/en/stable/) - For easy(er) image I/O
* [NumPy & SciPy](https://www.scipy.org/) - Matrix ops, Triangulation, Voronoi tesselation, etc.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
