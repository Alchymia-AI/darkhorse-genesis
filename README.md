<h1>🐎 Darkhorse Genesis</h1>
<p><strong>Experimental Mixture-of-Experts (MoE) Model</strong> | <strong>Research Lab:</strong> <a
        href="https://alchymia.ai" title="null">Alchymia AI</a></p>
<h2>🚀 Overview</h2>
<p><strong>Darkhorse Genesis</strong> is a high-efficiency "Large Language Model" (LLM) utilizing a proprietary sparsely
    activated <strong>Mixture-of-Experts (MoE)</strong> architecture. Optimized with <strong>Multi-Head Latent Attention
        (MLA)</strong>, it delivers frontier-level reasoning across specialized domains with minimal VRAM overhead.</p>
<h3>💎 Key Specifications</h3>
<ul>
    <li>
        <p><strong>Total Parameters:</strong> 5B (Scalable from 1B to 500B)</p>
    </li>
    <li>
        <p><strong>Active Parameters:</strong> ~1.2B per token (for 5B base)</p>
    </li>
    <li>
        <p><strong>Expert Count:</strong> 64 specialized experts across 10 domains</p>
    </li>
    <li>
        <p><strong>Context Window:</strong> 128K tokens (MLA Optimized)</p>
    </li>
    <li>
        <p><strong>Target Hardware:</strong> Consumer GPUs (12GB+ VRAM) or Apple Silicon (Metal)</p>
    </li>
</ul>
<h2>📊 Model Comparisons</h2>
<p>Darkhorse Genesis is designed to punch above its weight <span class="selected">class</span> by using sparse
    activation. Below is how it compares to standard dense LLMs.</p>
<table>
    <tbody>
        <tr>
            <th>
                <p>Metric</p>
            </th>
            <th>
                <p>Phi-3 Mini (Microsoft)</p>
            </th>
            <th>
                <p>Llama 3.2 3B (Meta)</p>
            </th>
            <th>
                <p><strong>Darkhorse Genesis</strong></p>
            </th>
        </tr>
        <tr>
            <td>
                <p><strong>Architecture</strong></p>
            </td>
            <td>
                <p>Dense Transformer</p>
            </td>
            <td>
                <p>Dense Transformer</p>
            </td>
            <td>
                <p><strong>MLA + Alchymia MoE</strong></p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>Total Params</strong></p>
            </td>
            <td>
                <p>3.8B</p>
            </td>
            <td>
                <p>3.2B</p>
            </td>
            <td>
                <p><strong>5B</strong></p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>Active Params</strong></p>
            </td>
            <td>
                <p>3.8B</p>
            </td>
            <td>
                <p>3.2B</p>
            </td>
            <td>
                <p><strong>~1.2B</strong></p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>Throughput</strong></p>
            </td>
            <td>
                <p>Baseline</p>
            </td>
            <td>
                <p>Baseline</p>
            </td>
            <td>
                <p><strong>~3x Faster</strong></p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>KV Cache</strong></p>
            </td>
            <td>
                <p>Standard</p>
            </td>
            <td>
                <p>Standard</p>
            </td>
            <td>
                <p><strong>Ultra-Low (MLA)</strong></p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>Specialization</strong></p>
            </td>
            <td>
                <p>Generalist</p>
            </td>
            <td>
                <p>Generalist</p>
            </td>
            <td>
                <p><strong>Domain-ID Isolated</strong></p>
            </td>
        </tr>
    </tbody>
</table>
<h2>🛠 Installation &amp; Setup</h2>
<h3>📋 Prerequisites</h3>
<ol>
    <li>
        <p><strong>Rust Toolchain:</strong> <a href="https://rustup.rs/" title="null">Install Rust</a></p>
    </li>
    <li>
        <p><strong>Libtorch:</strong> Required for <code>tch-rs</code>. <a href="https://pytorch.org/"
                title="null">Download PyTorch C++ Libs</a>.</p>
    </li>
</ol>
<p>&lt;!-- end list --&gt;</p>
<pre><code># Set environment variables for Libtorch (Linux/macOS)
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
<br class="ProseMirror-trailingBreak"></code></pre>
<h3>⚡ Quick Start</h3>
<pre><code>git clone [https://github.com/alchymia-ai/darkhorse-genesis.git](https://github.com/alchymia-ai/darkhorse-genesis.git)
cd darkhorse-genesis
cargo build --release
<br class="ProseMirror-trailingBreak"></code></pre>
<h2>🏗 Multi-Scale Configurations</h2>
<p>The engine now supports dynamic parameter scaling. Configuration files are located in the <code>parameters/</code>
    directory.</p>
<table>
    <tbody>
        <tr>
            <th>
                <p>Scale</p>
            </th>
            <th>
                <p>Command Flag</p>
            </th>
            <th>
                <p>Target Architecture</p>
            </th>
        </tr>
        <tr>
            <td>
                <p><strong>1B</strong></p>
            </td>
            <td>
                <p><code>--params 1b</code></p>
            </td>
            <td>
                <p>Edge Devices / Mobile</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>5B</strong></p>
            </td>
            <td>
                <p><code>--params 5b</code></p>
            </td>
            <td>
                <p><strong>Default</strong> - Consumer GPUs</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>30B+</strong></p>
            </td>
            <td>
                <p><code>--params 30b</code></p>
            </td>
            <td>
                <p>Multi-GPU Clusters</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>500B</strong></p>
            </td>
            <td>
                <p><code>--params 500b</code></p>
            </td>
            <td>
                <p>Frontier Research / HPC</p>
            </td>
        </tr>
    </tbody>
</table>
<h2>🧠 Inference</h2>
<p>Generate text directly via the Command Line Interface. The default profile uses a 512-hidden size configuration for
    rapid local execution.</p>
<pre><code># Direct CLI Inference with default settings
cargo run --release -- --prompt "Explain the laws of thermodynamics."
# Inference using a specific parameter scale
cargo run --release -- inference --params 1b --prompt "Write a Rust function."
<br class="ProseMirror-trailingBreak"></code></pre>

<div><h3>GGUF Support</h3><p>The <code>convert_to_gguf.py</code> script allows you to transform TorchScript/tch-rs models into GGUF format for use with <code>llama.cpp</code> or other GGUF-compatible backends.</p><h3>Features</h3><ul><li><p><strong>Auto-Detection</strong>: Automatically detects if the input is a standard <code>state_dict</code> or a <code>RecursiveScriptModule</code> (TorchScript).</p></li><li><p><strong>Architecture Mapping</strong>: Maps proprietary AlchymiaGen keys to standard GGUF tensor naming (MLA, MoE, and MTP support).</p></li><li><p><strong>Type Conversion</strong>: Standardizes weights to FP32 for broad compatibility.</p></li></ul><h3>Usage</h3><pre><code>python convert_hf_to_gguf.py  \
    --model-dir ./hf_model_out
<br class="ProseMirror-trailingBreak"></code></pre>
<p><strong>Running with llama.cpp:</strong>
To start the model using the <code>llama.cpp</code> main binary:</p>
<pre><code>./build/bin/llama-cli -m darkhorse-genesis.gguf -n 128 -p "The future of AI in emerging markets is"

<br class="ProseMirror-trailingBreak"></code></pre>
<h3>vLLM Support </h3>
<p>For high-throughput requirements, <code>darkhorse-genesis</code> is compatible with <strong>vLLM</strong> via the
    HuggingFace-standardized structure.</p>
<h3>Direct Loading</h3>
<p>vLLM can serve the model by pointing to the directory containing the <code>config.json</code> and the converted
    <code>.safetensors</code> or <code>.pt</code> weights.</p>
<pre><code>vllm serve /path/to/darkhorse-genesis \
    --tensor-parallel-size 1
<br class="ProseMirror-trailingBreak"></code></pre>
<h3>Optimized Attention (MLA)</h3>
<p>To leverage the <strong>Multi-Head Latent Attention (MLA)</strong> optimizations in vLLM:</p>
<ol>
    <li>
        <p>Ensure your <code>config.json</code> includes the <code>q_lora_rank</code> and <code>kv_lora_rank</code>
            parameters.</p>
    </li>
    <li>
        <p>Use vLLM version 0.6.0 or higher.</p>
    </li>
    <li>
        <p>The engine will automatically dispatch to optimized Triton kernels for latent attention.</p>
    </li>
</ol>
<h3>Mixture-of-Experts (MoE)</h3>
<p>For models utilizing MoE:</p>
<ul>
    <li>
        <p>vLLM utilizes <strong>Marlin</strong> or <strong>Megablocks</strong> kernels for efficient expert routing.
        </p>
    </li>
    <li>
        <p>If your weights are quantized, use the <code>--quantization</code> flag (e.g., <code>awq</code> or
            <code>fp8</code>).</p>
    </li>
</ul>
<h2>Technical Specifications</h2>
<table>
    <tbody>
        <tr>
            <th>
                <p>Feature</p>
            </th>
            <th>
                <p>GGUF Status</p>
            </th>
            <th>
                <p>vLLM Status</p>
            </th>
        </tr>
        <tr>
            <td>
                <p><strong>MLA Support</strong></p>
            </td>
            <td>
                <p>Yes (Linearized)</p>
            </td>
            <td>
                <p>Yes (Optimized Kernels)</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>MoE Routing</strong></p>
            </td>
            <td>
                <p>Static</p>
            </td>
            <td>
                <p>Dynamic</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>MTP Heads</strong></p>
            </td>
            <td>
                <p>Support via custom keys</p>
            </td>
            <td>
                <p>Supported via model-suffix</p>
            </td>
        </tr>
        <tr>
            <td>
                <p><strong>Quantization</strong></p>
            </td>
            <td>
                <p>Q4_K_M, Q5_K_M, etc.</p>
            </td>
            <td>
                <p>FP8, AWQ, GPTQ</p>
            </td>
        </tr>
    </tbody>
</table>
<h2>Troubleshooting</h2>
<p><strong>AttributeError: 'RecursiveScriptModule' object has no attribute 'items'</strong>
    This occurs when trying to treat a compiled TorchScript archive as a standard dictionary. Use the updated
    <code>convert_to_gguf.py</code> provided in this repo which handles JIT modules via <code>.state_dict()</code>
    extraction.</p>
<p><strong>UserWarning: torch.load received a zip file</strong>
    This is expected when loading <code>.ot</code> files created by Rust-based training pipelines. The converter handles
    this internally by dispatching to <code>torch.jit.load</code>.</p>
</div>

<h2>🧬 Data Pipeline &amp; Training</h2>
<p>Darkhorse Genesis requires structured data to train its specialized expert domains (Math, Code, Legal, etc.).</p>
<h3>1. Local Data Preparation</h3>
<p>Populate a <code>data/</code> directory with <code>.jsonl</code> files. Each entry should include a domain tag to
    assist the router.</p>
<p><strong>Format (<code>train_data.jsonl</code>):</strong></p>
<pre><code>{"domain": "code", "text": "fn main() { println!(\"Alchymia AI\"); }"}
{"domain": "math", "text": "The Riemann hypothesis is a conjecture that..."}
<br class="ProseMirror-trailingBreak"></code></pre>



<h3>2. Training Execution</h3>
<p>To initiate training with a specific parameter scale and the <strong>Auxiliary-Loss-Free (ALF)</strong> balancing
    strategy:</p>
<pre><code># Train a 1B parameter model
cargo run --release -- train --params 1b --data ./data/train_data.jsonl

cargo run --release -- train --params 5b --data ./data/train_data.jsonl


<br class="ProseMirror-trailingBreak"></code></pre>
<h2>🚢 Production Deployment</h2>
<h3>🐳 Containerization (Docker)</h3>
<pre><code>FROM nvidia/cuda:12.1.0-base-ubuntu22.04

LABEL maintainer="Alchymia AI"
COPY ./target/release/darkhorse-genesis /usr/local/bin/

ENTRYPOINT ["darkhorse-genesis"]
<br class="ProseMirror-trailingBreak"></code></pre>
<h2>📄 License</h2>
<p>Copyright © 2024 <strong>Alchymia AI</strong>.</p>
<blockquote>
    <p>[!IMPORTANT]
        For technical inquiries, please refer to the <a href="" title="null">Darkhorse Genesis Release Paper</a>.</p>
</blockquote>