/// Multi-Token Prediction (MTP) Module
/// Alchymia AI Research Lab | Lead Architect: Lifeofanisland
///
/// MTP enables the model to predict n+k tokens simultaneously,
/// improving planning capabilities and enabling speculative decoding.
use crate::config::Config;
use tch::{nn, nn::Module, Tensor};

/// The MTP module contains projection heads for future token prediction.
pub struct MTP {
    /// Linear heads for predicting subsequent tokens beyond n+1
    heads: Vec<nn::Linear>,
}

impl MTP {
    /// Constructs the MTP module based on the Config's `num_next_tokens`.
    pub fn new(vs: &nn::Path, config: &Config) -> Self {
        let mut heads = Vec::new();
        let heads_path = vs / "heads";

        for i in 0..config.num_next_tokens {
            heads.push(nn::linear(
                &heads_path / i,
                config.hidden_size,
                config.hidden_size,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ));
        }

        Self { heads }
    }

    /// Transforms the final hidden state into multiple future token representations.
    pub fn forward(&self, hidden_state: &Tensor) -> Vec<Tensor> {
        self.heads
            .iter()
            .map(|head| hidden_state.apply(head))
            .collect()
    }
}
