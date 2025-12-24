/// Darkhorse (Genesis) Configuration
/// Developed by Alchymia AI | Lead Architect: Lifeofanisland
///
/// This module centralizes the architectural hyperparameters and
/// model metadata for the 5B LLM profile.
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_name: String,
    pub version: String,
    pub lab: String,
    pub author: String,

    // Network Dimensions
    pub hidden_size: i64, // The latent width of the model (3072 for 5B)
    pub num_heads: i64,   // Attention head count for MLA
    pub head_dim: i64,    // Dimension per head (128 recommended)
    pub n_layers: i64,    // Total transformer depth

    // Compression Parameters
    pub kv_lora_rank: i64, // Rank for KV latent compression
    pub q_lora_rank: i64,  // Rank for Query latent compression

    // MoE Topology
    pub moe_num_experts: i64, // Total expert pool (64 for Genesis)
    pub moe_top_k: i64,       // Experts activated per token (4 for ~1.2B active)
    pub num_next_tokens: i64, // MTP signal count

    pub vocab_size: i64, // Vocabulary and Training
}

impl Config {
    /// Load configuration from a JSON file in the parameters/ directory
    pub fn from_file(param_scale: &str) -> anyhow::Result<Self> {
        let path_str = format!("parameters/{}.json", param_scale);
        let path = Path::new(&path_str);

        if !path.exists() {
            return Err(anyhow::anyhow!(
                "Configuration for {} not found at {:?}",
                param_scale,
                path
            ));
        }

        let json_str = fs::read_to_string(path)?;
        let static_json: &'static str = Box::leak(json_str.into_boxed_str());
        let config: Config = serde_json::from_str(static_json)?;
        Ok(config)
    }
}

impl Default for Config {
    // Defaults for the model profile.
    fn default() -> Self {
        Self {
            model_name: "Darkhorse".to_string(),
            version: "Genesis".to_string(),
            lab: "Alchymia AI".to_string(),
            author: "Lifeofanisland".to_string(),

            hidden_size: 512, // Lower from 3072
            num_heads: 8,     // Lower from 32
            head_dim: 64,     // Lower from 128
            n_layers: 6,      // Lower from 32 (huge impact on memory)

            kv_lora_rank: 64,   // Lower from 156
            q_lora_rank: 128,   // Lower from 768
            moe_num_experts: 8, // Lower from 64
            moe_top_k: 2,       // Lower from 4
            num_next_tokens: 1,
            vocab_size: 32768, // 32768 is a standard efficient size for local testing (Prod: 102400)
        }
    }
}
