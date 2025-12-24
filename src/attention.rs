use crate::config::Config;
use tch::{nn, nn::Module, Tensor, Kind};

pub struct MLA {
    kv_down_proj: nn::Linear,
    kv_up_proj: nn::Linear,
    q_down_proj: nn::Linear,
    q_up_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: i64,
    head_dim: i64,
}

impl MLA {
    pub fn new(vs: &nn::Path, config: &Config) -> Self {
        let kv_rank = config.kv_lora_rank;
        let q_rank = config.q_lora_rank;
        let full_kv_dim = config.num_heads * config.head_dim * 2;
        
        Self {
            kv_down_proj: nn::linear(vs / "kv_down", config.hidden_size, kv_rank, Default::default()),
            kv_up_proj: nn::linear(vs / "kv_up", kv_rank, full_kv_dim, Default::default()),
            q_down_proj: nn::linear(vs / "q_down", config.hidden_size, q_rank, Default::default()),
            q_up_proj: nn::linear(vs / "q_up", q_rank, config.num_heads * config.head_dim, Default::default()),
            out_proj: nn::linear(vs / "out_proj", config.num_heads * config.head_dim, config.hidden_size, Default::default()),
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Fix: size3() returns a tuple (i64, i64, i64), use () instead of []
        let (b, s, _) = x.size3().expect("Input must be a 3D tensor [Batch, Seq, Hidden]");

        // KV Latent Compression
        let latent_kv = x.apply(&self.kv_down_proj);
        let kv = latent_kv.apply(&self.kv_up_proj)
            .view([b, s, self.num_heads, 2 * self.head_dim]);
        
        let k = kv.narrow(-1, 0, self.head_dim).transpose(1, 2);
        let v = kv.narrow(-1, self.head_dim, self.head_dim).transpose(1, 2);

        // Q Latent Compression
        let q = x.apply(&self.q_down_proj)
            .apply(&self.q_up_proj)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let scores = (q.matmul(&k.transpose(-2, -1))) / (self.head_dim as f64).sqrt();
        let attn = scores.softmax(-1, Kind::Float);
        
        let context = attn.matmul(&v)
            .transpose(1, 2)
            .reshape(&[b, s, self.num_heads * self.head_dim]);

        context.apply(&self.out_proj)
    }
}