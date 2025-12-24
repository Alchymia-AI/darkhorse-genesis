/// AlchymiaGen Weight Exporter
/// Alchymia AI Research Lab | Lead Architect: Lifeofanisland
///
/// This script iterates through the VarStore and saves tensors to a format
/// that can be easily ingested by the Safetensors converter.

use tch::{nn, nn::VarStore, Device};
use std::path::Path;

pub fn export_to_safetensors_raw(vs: &VarStore, export_path: &Path) -> anyhow::Result<()> {
    println!("--- 🐎 AlchymiaGen Weight Export ---");
    
    // We utilize the named_variables() iterator to capture the exact
    // hierarchy defined in model.rs, moe.rs, and attention.rs
    let variables = vs.variables();
    
    // In tch-rs, we can save the entire VarStore as a named-tensor file
    // which Python can read via torch.load or a custom parser.
    vs.save(export_path)?;
    
    println!("Raw weights exported to: {:?}", export_path);
    println!("Total tensors exported: {}", variables.len());
    
    Ok(())
}