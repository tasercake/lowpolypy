# LowPoly

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
