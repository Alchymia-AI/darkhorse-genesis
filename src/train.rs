/// Alchymia AI Training Engine
/// Developed by Alchymia AI | Lead Architect: Lifeofanisland
/// 
/// This module implements the training loop for Darkhorse Genesis, 
/// focusing on the ALF (Auxiliary-Loss-Free) balancing and MTP loss.

use crate::config::Config;
use crate::model::AlchymiaGen;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use std::time::Instant;
use std::path::Path;
use std::process::Command;

pub struct Trainer {
    pub device: Device,
    pub batch_size: i64,
    pub epochs: i64,
    pub lr: f64,
    pub config: Config,
}

impl Trainer {
    pub fn new(config: &Config, device: Device) -> Self {
        
        Self {
            device,
            batch_size: 32,
            epochs: 3,
            lr: 1e-4,
            config: config.clone(),
        }
    }

    pub fn train(&self, vs: &mut nn::VarStore, vocab_size: i64) -> anyhow::Result<()> {
        // Ensure the VarStore is moved to the detected device (MPS/Cuda)
        vs.set_device(self.device);
        
        let model = AlchymiaGen::new(&vs.root(), &self.config, vocab_size);
        
        let output_head = nn::linear(
            &vs.root() / "output_head",
            self.config.hidden_size,
            vocab_size,
            nn::LinearConfig { bias: false, ..Default::default() }
        );

        let mut opt = nn::Adam::default().build(vs, self.lr)?;

        println!("Starting training on {:?}", self.device);
        let seq_len = 128;

        for epoch in 1..=self.epochs {
            let start = Instant::now();
            let mut total_loss = 0.0f32;

            for step in 0..100 {
                opt.zero_grad();

                let input = Tensor::randint(vocab_size, &[self.batch_size, seq_len], (Kind::Int, self.device));
                let targets = Tensor::randint(vocab_size, &[self.batch_size, seq_len], (Kind::Int, self.device));

                let (hidden_states, mtp_hidden_states) = model.forward(&input);

                let logits = hidden_states.apply(&output_head);
                let flattened_logits = logits.view([-1, vocab_size]);
                let flattened_targets = targets.view([-1]).to_kind(Kind::Int64);

                let main_loss = flattened_logits.cross_entropy_for_logits(&flattened_targets);

                let mut mtp_loss = Tensor::from(0.0f32).to_device(self.device);
                
                for mtp_hidden in mtp_hidden_states {
                    let mtp_logits = mtp_hidden.apply(&output_head);
                    let m_loss = mtp_logits.view([-1, vocab_size]).cross_entropy_for_logits(&flattened_targets);
                    mtp_loss = mtp_loss + m_loss;
                }

                let mtp_weight = Tensor::from(0.1f32).to_device(self.device);
                let loss = &main_loss + (&mtp_weight * &mtp_loss);
                
                loss.backward();
                opt.step();

                let current_loss = loss.to_kind(Kind::Float).double_value(&[]) as f32;
                total_loss += current_loss;
                
                if step % 20 == 0 {
                    println!("  Step: {} | Loss: {:.4}", step, current_loss);
                }
            }

            let duration = start.elapsed();
            let avg_loss = total_loss / 100.0;
            
            println!(
                "Epoch: [{}/{}] | Avg Loss: {:.4} | Duration: {:?}",
                epoch, self.epochs, avg_loss, duration
            );

            // save checkpoint
            let checkpoint_name = format!("checkpoint_epoch_{}.ot", epoch);
            println!("Saving checkpoint to {}...", checkpoint_name);
            vs.save(Path::new(&checkpoint_name))?;
        }

        // save model locally
        let final_model_path = "model.ot";
        println!("Training complete. Saving final model to {}", final_model_path);
        vs.save(Path::new("model.ot"))?;

        // auto conversion t0 huggingface
        self.initiate_hf_conversion(final_model_path);

        Ok(())
    }

    /// Triggers the external Python conversion script
    fn initiate_hf_conversion(&self, model_path: &str) {

        let script_path = "convert_to_hf.py";
        
        if !Path::new(script_path).exists() {
            println!("Warning: {} not found. Skipping auto-conversion.", script_path);
            return;
        }

        println!("Initiating HuggingFace Safetensors conversion...");
        
        let output = Command::new("python3")
            .arg(script_path)
            .arg(model_path) // Passing the model path as an argument
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    println!("✅ Conversion successful: {}", String::from_utf8_lossy(&out.stdout));
                } else {
                    eprintln!("❌ Conversion failed: {}", String::from_utf8_lossy(&out.stderr));
                }
            },
            Err(e) => eprintln!("❌ Failed to execute conversion script: {}", e),
        }
    }

}