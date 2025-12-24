/// AlchymiaGen (1.0-gen) Training Binary
/// Alchymia AI Research Lab 

mod config;
mod model;
mod attention;
mod moe;
mod mtp;
mod train;

use crate::config::Config;
use crate::train::Trainer;
use tch::nn;

fn main() -> anyhow::Result<()> {
    // 1. Initialize Configuration
    let config = Config::default();
    let vocab_size = 102400; // AlchymiaGen vocabulary scale

    // 2. Initialize VarStore (Holds the trainable parameters)
    let mut vs = nn::VarStore::new(tch::Device::cuda_if_available());

    // 3. Initialize Trainer
    let trainer = Trainer::new(&config);

    // 4. Start Training Loop
    trainer.train(&mut vs, vocab_size)?;

    Ok(())
}