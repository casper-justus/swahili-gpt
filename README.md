# Swahili GPT (MiniGPT-JAX) 🇰🇪🤖

A ~30 million parameter language model trained entirely from scratch on a pure Kiswahili dataset. Built using JAX and Flax (NNX API), this project demonstrates an end-to-end pipeline for training a compute-optimal Small Language Model (SLM) on Google Colab's free T4 GPUs.

## 🌟 Project Overview

This model is designed to be a highly efficient, lightweight Kiswahili text generation engine. By restricting the vocabulary to 10,000 tokens, the BPE tokenizer is forced to learn the agglutinative grammatical structure of Kiswahili (prefixes, infixes, roots) rather than memorizing massive words.

### Model Architecture
- **Parameters:** ~30 Million
- **Framework:** JAX / Flax (NNX API) / Optax
- **Context Window:** 1024 tokens
- **Vocabulary Size:** 10,000 tokens (Custom BPE Tokenizer)
- **Embedding Dimension:** 512
- **Attention Heads:** 8
- **Transformer Blocks:** 6
- **Training Target:** 200,000 steps (Chinchilla compute-optimal ratio)

## 📂 Repository Structure

- `MiniGPT_Kiswahili_Resumable_Final.ipynb`: The core training notebook. Features Google Drive integration for multi-day resumable training and auto-checkpointing via Orbax.
- `kenya_tokenizer.json`: The custom 10k vocabulary BPE tokenizer trained on Kiswahili text.
- `inference.py` *(Coming soon)*: Script to load the Orbax checkpoints and generate text locally.

## 🚀 Getting Started

To train this model yourself:

1. Open `MiniGPT_Kiswahili_Resumable_Final.ipynb` in Google Colab.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Upload `kenya_tokenizer.json` to your Colab workspace.
4. Run all cells. The script will automatically mount your Google Drive and save/resume checkpoints every 500 steps.

## 📊 Dataset
The model is trained on the `marcoharuni95/swahili-text-corpus` (via Hugging Face), a modern, deduplicated Parquet dataset of unannotated Kiswahili text.

## 💡 Future Plans
- Exporting final `.safetensors` weights.
- Deploying a live Gradio web demo on Hugging Face Spaces.
- Building a custom React/Next.js frontend to interface with the model.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
