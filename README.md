# Swahili GPT (MiniGPT-JAX) 🇰🇪🤖

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casper-justus/swahili-gpt/blob/main/MiniGPT_Kiswahili_Resumable_Final.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![JAX](https://img.shields.io/badge/Framework-JAX%20%2B%20Flax-blue)
![Language](https://img.shields.io/badge/Language-Kiswahili-green)

A ~30 million parameter language model trained entirely from scratch on a pure Kiswahili dataset. Built using JAX and Flax (NNX API), this project demonstrates an end-to-end pipeline for training a compute-optimal Small Language Model (SLM) on Google Colab's free T4 GPUs.

## 📉 Training Loss Curve

![Training Loss Curve](assets/loss_curve.png)

> Loss dropped from **9.56 → ~3.8** in the first 5,000 steps, confirming the model is learning Kiswahili structure. Full training runs to 200,000 steps.

## 🌟 Project Overview

This model is designed to be a highly efficient, lightweight Kiswahili text generation engine. By restricting the vocabulary to 10,000 tokens, the BPE tokenizer is forced to learn the agglutinative grammatical structure of Kiswahili (prefixes, infixes, roots) rather than memorizing massive words.

### Model Architecture
| Hyperparameter | Value |
|---|---|
| Parameters | ~30 Million |
| Framework | JAX / Flax (NNX API) / Optax |
| Context Window | 1024 tokens (~750 Swahili words) |
| Vocabulary Size | 10,000 tokens (Custom BPE) |
| Embedding Dim | 512 |
| Attention Heads | 8 |
| Transformer Blocks | 6 |
| Training Target | 200,000 steps (Chinchilla-optimal) |

## 📂 Repository Structure

```
swahili-gpt/
├── MiniGPT_Kiswahili_Resumable_Final.ipynb  # Training notebook (Google Colab)
├── inference.py                              # Generate text from a saved checkpoint
├── requirements.txt                          # Python dependencies
├── kenya_tokenizer.json                      # Custom 10k BPE tokenizer (upload to Drive)
├── assets/
│   └── loss_curve.png                        # Training loss chart
└── LICENSE
```

## 🚀 Getting Started

### Train the model (Google Colab)
1. Click the **Open in Colab** badge above.
2. Go to **Runtime → Change runtime type → T4 GPU → Save**.
3. Upload `kenya_tokenizer.json` to your Google Drive at `/MyDrive/MiniGPT_Kiswahili_Checkpoints/`.
4. Run all cells. Checkpoints save to Drive every 5,000 steps — reconnect anytime to resume.

### Generate text locally (after training)
```bash
# 1. Clone the repo
git clone https://github.com/casper-justus/swahili-gpt.git
cd swahili-gpt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference
python inference.py \
  --ckpt_dir ./MiniGPT_Kiswahili_Checkpoints \
  --prompt "Habari za asubuhi" \
  --steps 100 \
  --temperature 0.8
```

## 📊 Dataset
Trained on [`marcoharuni95/swahili-text-corpus`](https://huggingface.co/datasets/marcoharuni95/swahili-text-corpus) — a modern, deduplicated Parquet-format Kiswahili corpus (~17.6M tokens) loaded directly from Hugging Face.

## 💡 Future Plans
- [ ] Export final weights to `.safetensors` format.
- [ ] Deploy a live Gradio demo on Hugging Face Spaces.
- [ ] Build a custom React/Next.js frontend for the demo.

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
