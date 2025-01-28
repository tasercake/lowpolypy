# LowPolyPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/maschere/6c789d70bbdaed2d89e1742f9d50a508/lowpolypy.ipynb)

Generate Low-Poly Versions of Images:

![Samples](images/samples.jpg)

## Usage

```shell
cargo run --release -p lowpoly_cli ./images/bird1.jpg
```

## Built With

- [image](https://opencv.org/releases/) - Image manipulation library
- [imageproc](https://shapely.readthedocs.io/) - For geometry operations

- [OpenAI o1](https://openai.com/o1/) — Helped me figure out the trickly geometry algorithms
