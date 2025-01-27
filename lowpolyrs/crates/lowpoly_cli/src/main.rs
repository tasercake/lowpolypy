use clap::{arg, command, Parser};
use env_logger;
use std::path::PathBuf;

use lowpoly::to_lowpoly;

/// A simple CLI definition using `clap`.
#[derive(Parser, Debug)]
#[command(
    name = "lowpolypy",
    author,
    version,
    about = "Generate low-poly versions of images"
)]
pub struct Cli {
    /// Path to source image.
    pub source: PathBuf,

    /// Path to destination directory (optional). If omitted, we infer
    /// a directory named 'lowpoly' next to the source image.
    #[arg(short, long)]
    pub destination: Option<PathBuf>,

    /// Number of anchor points to sample from the image.
    #[arg(long, default_value_t = 1000)]
    pub num_points: u32,

    /// Final output size in pixels.
    #[arg(long, default_value_t = 2560)]
    pub output_size: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    let source = cli.source;
    let destination = cli.destination;
    let num_points = cli.num_points;
    let output_size = cli.output_size;

    // Run the lowpoly transformation
    to_lowpoly(source, destination, num_points, output_size)
        // Handle errors
        .map_err(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        })
}
