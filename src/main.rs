mod attention;
/// Alchymia Darkhorse (Genesus) Main Entry Point
/// Alchymia AI Research Lab 
///
/// This file serves as the unified entry point for both real inference
/// and training modes for the AlchymiaGen model, featuring production CLI handling.
mod config;
mod inference;
mod model;
mod moe;
mod mtp;
mod train;

use crate::config::Config;
use crate::inference::run_inference;
use crate::train::Trainer;
use clap::{Parser, Subcommand};
use tch::{nn, Device};

#[derive(Parser)]
#[command(name = "darkhorse-genesis")]
#[command(about = "Alchymia AI 5B MoE Inference & Training Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The prompt to generate text from (shortcut for inference)
    #[arg(short, long)]
    prompt: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the training loop with ALF balancing
    Train {
        /// Path to the training data
        #[arg(short, long, default_value = "data/train.jsonl")]
        data: String,

        /// Parameter configuration to use (e.g., 1b, 5b, 30b)
        #[arg(short, long, default_value = "5b")]
        params: String,
    },
    /// Run model inference
    Inference {
        /// The prompt for the model
        #[arg(short, long)]
        prompt: String,

        #[arg(short, long, default_value_t = 128)]
        max_tokens: i64,

        /// Parameter configuration to use (e.g., 1b, 5b)
        #[arg(short, long, default_value = "5b")]
        params: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        println!("Apple Silicon GPU (MPS) detected. Enabling Metal acceleration.");
        Device::Mps
    } else {
        println!("No GPU detected. Falling back to CPU.");
        Device::Cpu
    };

    let mut vs = nn::VarStore::new(device);

    match cli.command {
        Some(Commands::Train { data, params }) => {
            println!("🚀 [Alchymia AI] Initializing Training Mode");
            println!("Using Parameter Set: {}", params);
            println!("Dataset: {}", data);

            let config = Config::from_file(&params)?;
            let trainer = Trainer::new(&config, device);
            trainer.train(&mut vs, config.vocab_size)?;
        }
        Some(Commands::Inference {
            prompt,
            max_tokens,
            params,
        }) => {
            let config = Config::from_file(&params)?;
            run_inference(
                &mut vs,
                &config,
                config.vocab_size,
                device,
                &prompt,
                max_tokens,
            )?;
        }
        None => {
            if let Some(prompt) = cli.prompt {
                let config = Config::default();
                run_inference(&mut vs, &config, config.vocab_size, device, &prompt, 128)?;
            } else {
                println!("Usage: cargo run -- --prompt \"Your text here\" or use the 'inference' subcommand.");
            }
        }
    }

    Ok(())
}
