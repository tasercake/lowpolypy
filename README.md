# LowPolyPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/maschere/6c789d70bbdaed2d89e1742f9d50a508/lowpolypy.ipynb)

Generate Low-Poly Versions of Images:

![Samples](images/samples.jpg)

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

```shell
git clone https://github.com/tasercake/lowpolypy
cd lowpolypy
```

### Installing

Install dependencies:

```shell
uv sync
```

### Try It Out

From the root of the project directory, run

```shell
lowpolypy './images/giraffe.jpg'
```

This should create a new image in the `images` directory.

## Run the server

If you'd like to run the server locally, install the `server` dependency group:

```shell
uv sync --group server
```

Then serve the FastAPI app with uvicorn:

```shell
uvicorn lowpolypy.server:app --reload
```

## Built With

- [OpenCV](https://opencv.org/releases/) - Image manipulation library
- [Shapely](https://shapely.readthedocs.io/) - For geometry operations
- [NumPy & SciPy](https://www.scipy.org/) - Matrix ops, Triangulation, Voronoi tesselation, etc.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
