/// Darkhorse Genesis MoE Implementation
/// Alchymia AI Research Lab | Lifeofanisland
///
/// Specialized Domain Experts with ALF (Auxiliary-Loss-Free) Balancing.
use crate::config::Config;
use tch::{nn, nn::Module, IndexOp, Kind, Tensor};

/// Specialized Expert Domains for the Alchymia Network
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    General,
    Mathematics,
    Code,
    Physics,
    Business,
    Legal,
    Government,
    Biomedical,
    Logic,
    Creative,
}

/// Expert Unit: Production SwiGLU implementation for high-throughput inference.
#[derive(Debug)]
pub struct Expert {
    pub id: i64,
    pub domain: Domain,
    w1: nn::Linear, // Gating projection
    w2: nn::Linear, // Output projection
    w3: nn::Linear, // Upscale projection
}

impl Expert {
    fn new(vs: &nn::Path, id: i64, domain: Domain, h: i64, inter: i64) -> Self {
        let cfg = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            id,
            domain,
            w1: nn::linear(vs / "w1", h, inter, cfg),
            w2: nn::linear(vs / "w2", inter, h, cfg),
            w3: nn::linear(vs / "w3", h, inter, cfg),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = x.apply(&self.w1).silu();
        let up = x.apply(&self.w3);
        (gate * up).apply(&self.w2)
    }
}

pub struct AlchymiaMoE {
    gate_weight: Tensor,
    bias: Tensor,
    experts: Vec<Expert>,
    shared_expert: Expert,
    top_k: i64,
}

impl AlchymiaMoE {
    pub fn new(vs: &nn::Path, config: &Config) -> Self {
        let inter = (4 * config.hidden_size * 2) / 3;
        let domains = [
            Domain::Mathematics,
            Domain::Code,
            Domain::Physics,
            Domain::Business,
            Domain::Legal,
            Domain::Government,
            Domain::Biomedical,
            Domain::Logic,
            Domain::Creative,
            Domain::General,
        ];

        let mut experts = Vec::new();
        for i in 0..config.moe_num_experts {
            let domain = domains[(i as usize) % domains.len()];
            experts.push(Expert::new(
                &((vs / "experts") / i),
                i,
                domain,
                config.hidden_size,
                inter,
            ));
        }

        // Fix: Standard Uniform initialization variant for weight variables.
        // Calculate the Kaiming Uniform bounds (1/sqrt(fan_in)) manually to ensure
        // type compatibility with the vs.var() expected Init enum.
        let bound = 1.0 / (config.hidden_size as f64).sqrt();

        // Don't remove yet
        /*
               let init_kv = nn::Init::Kaiming {
                   dist: Init::Uniform {
                       lo: -1.0 / (config.hidden_size as f64).sqrt(),
                       up: 1.0 / (config.hidden_size as f64).sqrt(),
                   },
                   fan: FanInOut::FanIn,
                   non_linearity: NonLinearity::LeakyRelu(5.0f64.sqrt()),
               };
        */
        let init_kv = nn::Init::Uniform {
            lo: -bound,
            up: bound,
        };

        Self {
            gate_weight: vs.var(
                "gate_weight",
                &[config.moe_num_experts, config.hidden_size],
                init_kv,
            ),
            bias: vs.zeros("routing_bias", &[config.moe_num_experts]),
            experts,
            shared_expert: Expert::new(
                &(vs / "shared"),
                -1,
                Domain::General,
                config.hidden_size,
                inter,
            ),
            top_k: config.moe_top_k,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let size = x.size();
        let (b, s, h) = (size[0], size[1], size[2]);

        let shared_out = self.shared_expert.forward(x);
        let routing_logits = x.matmul(&self.gate_weight.tr());
        let scores = routing_logits + &self.bias;

        let (top_weights, top_idx) = scores
            .softmax(-1, Kind::Float)
            .topk(self.top_k, -1, true, true);

        // Explicit slice type for IntListOption trait compliance
        let norm_weights =
            &top_weights / top_weights.sum_dim_intlist(&[-1i64][..], true, Kind::Float);

        let x_flat = x.view([-1, h]);
        let top_idx_flat = top_idx.view([-1, self.top_k]);
        let top_weights_flat = norm_weights.view([-1, self.top_k]);

        let mut combined_output = Tensor::zeros_like(&x_flat);

        for k in 0..self.top_k {
            let current_indices = top_idx_flat.i((.., k));
            let current_weights = top_weights_flat.i((.., k)).unsqueeze(-1);

            for (i, expert) in self.experts.iter().enumerate() {
                let mask = current_indices.eq(i as i64);

                if mask.any().int64_value(&[]) != 0 {
                    let indices = mask.nonzero().flatten(0, -1);
                    let tokens = x_flat.index_select(0, &indices);
                    let out = expert.forward(&tokens);
                    let weighted_out = out * current_weights.index_select(0, &indices);
                    combined_output = combined_output.index_add(0, &indices, &weighted_out);
                }
            }
        }

        shared_out + combined_output.view([b, s, h])
    }
}
