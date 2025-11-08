use std::{env, path::Path};

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .context("usage: cargo run -p yunet-core --example inspect_node <model.onnx> [node_id]")?;
    let node_filter = args
        .next()
        .map(|value| value.parse::<usize>().context("invalid node id"));
    let node_filter = match node_filter.transpose()? {
        Some(id) => Some(id),
        None => None,
    };

    let model = tract_onnx::onnx()
        .model_for_path(Path::new(&model_path))
        .with_context(|| format!("failed to parse model {model_path}"))?;

    println!("Model: {}", model_path);
    println!("Inputs:");
    for input in model.inputs.iter() {
        let fact = model.outlet_fact(*input)?;
        println!("  #{:?}: {:?}", input.node, fact);
    }

    println!("Nodes:");
    for (idx, node) in model.nodes().iter().enumerate() {
        if node_filter.map(|target| target == idx).unwrap_or(true) {
            println!(
                "#{} \"{}\" op={} inputs={} outputs={}",
                idx,
                node.name,
                node.op.name(),
                node.inputs.len(),
                node.outputs.len()
            );
            for (slot, input) in node.inputs.iter().enumerate() {
                let fact = model.outlet_fact(*input)?;
                println!("    input[{slot}] <- {:?}: {:?}", input, fact);
            }
            for (slot, _output) in node.outputs.iter().enumerate() {
                if let Ok(fact) = model.outlet_fact(OutletId::new(idx, slot)) {
                    println!("    output[{slot}] -> {:?}", fact);
                }
            }
        }
    }

    println!("Attempting to type-check...");
    match model.clone().into_typed() {
        Ok(_) => println!("Typed conversion succeeded."),
        Err(err) => {
            println!("Typed conversion failed: {err}");
            for (i, cause) in err.chain().enumerate() {
                println!("  cause[{i}]: {cause}");
            }
        }
    }

    Ok(())
}
