# LowPolyPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/maschere/6c789d70bbdaed2d89e1742f9d50a508/lowpolypy.ipynb)

Generate Low-Poly Versions of Images:

![Samples](images/samples.jpg)

## Usage

```shell
cargo run --release -p lowpoly_cli ./images/bird.jpg
```

## Built With

- [image](https://crates.io/crates/image) - Basic image processing library
- [imageproc](https://crates.io/crates/imageproc) - For more advanced image processing & pixel ops

- [OpenAI o1](https://openai.com/o1/) — Helped me figure out the trickly geometry algorithms
