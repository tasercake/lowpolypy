use clap::{arg, command, Parser};
use env_logger;
use image::ImageFormat;
use log::info;
use std::{fs, path::PathBuf};

use lowpoly::file_utils::{parse_or_infer_destination, validate_image_source};
use lowpoly::to_lowpoly;

/// A simple CLI definition using `clap`.
#[derive(Parser, Debug)]
#[command(
    name = "lowpoly",
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

    /// The emphasis placed on edges in the images. Higher values make the edges more prominent. Default is 2.2.
    #[arg(long, default_value_t = 2.2)]
    pub edge_focus: f32,

    /// Number of random filler points to sample
    #[arg(long, default_value_t)]
    pub num_random_points: u32,

    /// Final output size in pixels.
    #[arg(long, default_value_t = 2560)]
    pub output_size: u32,

    /// Draw and save debug images
    #[arg(long, default_value_t = false)]
    pub debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    let source = cli.source;
    let destination = cli.destination;
    let num_points = cli.num_points;
    let sharpness = cli.edge_focus;
    let num_random_points = cli.num_random_points;
    let output_size = cli.output_size;
    let debug = cli.debug;

    // Validate & expand paths
    let source = validate_image_source(&source)?;
    let destination_dir = parse_or_infer_destination(&destination, &source, "lowpoly")?;
    fs::create_dir_all(&destination_dir)?;

    // Construct output filename
    let source_stem = source
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let output_path = destination_dir.join(format!("{}.png", source_stem));

    // Load image
    info!("Processing {} ...", source.display());
    let image = image::open(&source)
        .map_err(|e| format!("Failed to read image {}: {}", source.display(), e))?;

    // Run the lowpoly transformation
    let result = to_lowpoly(
        &image,
        num_points,
        sharpness,
        num_random_points,
        output_size,
        debug,
    )
    .map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });

    // Save the output
    info!("Writing output to: {}", output_path.display());
    result
        .as_ref()
        .unwrap()
        .lowpoly_image
        .save_with_format(&output_path, ImageFormat::Png)?;

    // Save the debug images with suffix "_debug{index}"
    for (i, debug_image) in result.as_ref().unwrap().debug_images.iter().enumerate() {
        let debug_path = destination_dir.join(format!("{}_debug{}.png", source_stem, i));
        if let Some(debug_image) = debug_image {
            debug_image.save_with_format(&debug_path, ImageFormat::Png)?;
        }
    }

    Ok(())
}
