/// AlchymiaGen (1.0-gen) Transformer Architecture
/// Alchymia AI Research Lab 
/// 
/// This module implements the main Transformer block, integrating 
/// Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE).

use crate::config::Config;
use crate::attention::MLA;
use crate::moe::AlchymiaMoE;
use crate::mtp::MTP;
use tch::{nn, nn::Module, Tensor};

/// The primary AlchymiaGen model structure.
pub struct AlchymiaGen {
    embed: nn::Embedding,
    layers: Vec<AlchymiaLayer>,
    norm: nn::LayerNorm,
    mtp: MTP,
}

/// A single Transformer layer combining MLA and MoE.
struct AlchymiaLayer {
    attention: MLA,
    attn_norm: nn::LayerNorm,
    moe: AlchymiaMoE,
    moe_norm: nn::LayerNorm,
}

impl AlchymiaGen {
    /// Initializes the AlchymiaGen model with the specified configuration.
    pub fn new(vs: &nn::Path, config: &Config, vocab_size: i64) -> Self {
        let mut layers = Vec::new();
        let layers_path = vs / "layers";

        for i in 0..config.n_layers {
            let layer_path = &layers_path / i;
            
            // Fix: In tch-rs, the '/' operator moves the path. 
            // Use the '&' reference to create sub-paths without moving 'layer_path'.
            layers.push(AlchymiaLayer {
                attention: MLA::new(&(&layer_path / "attention"), config),
                attn_norm: nn::layer_norm(&(&layer_path / "attn_norm"), vec![config.hidden_size], Default::default()),
                moe: AlchymiaMoE::new(&(&layer_path / "moe"), config),
                moe_norm: nn::layer_norm(&(&layer_path / "moe_norm"), vec![config.hidden_size], Default::default()),
            });
        }

        Self {
            embed: nn::embedding(vs / "embed", vocab_size, config.hidden_size, Default::default()),
            layers,
            norm: nn::layer_norm(vs / "norm", vec![config.hidden_size], Default::default()),
            mtp: MTP::new(&(vs / "mtp"), config),
        }
    }

    /// Executing the forward pass across all layers with residual connections.
    pub fn forward(&self, tokens: &Tensor) -> (Tensor, Vec<Tensor>) {
        let mut x = tokens.apply(&self.embed);

        for layer in &self.layers {
            // 1. Attention Block with Pre-Norm Residual
            let attn_residual = &x;
            let x_norm = x.apply(&layer.attn_norm);
            x = attn_residual + layer.attention.forward(&x_norm);

            // 2. MoE Block with Pre-Norm Residual
            let moe_residual = &x;
            let x_norm = x.apply(&layer.moe_norm);
            x = moe_residual + layer.moe.forward(&x_norm);
        }

        let final_hidden = x.apply(&self.norm);
        
        // Compute Multi-Token Predictions for denser training signal
        let mtp_predictions = self.mtp.forward(&final_hidden);

        (final_hidden, mtp_predictions)
    }
}