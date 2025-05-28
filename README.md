# LLM Project: Building a Large Language Model From Scratch

This repository contains an end-to-end implementation of a Large Language Model (LLM) pipeline, inspired by Sebastian Raschka's book ["Build a Large Language Model From Scratch"](https://www.manning.com/books/build-a-large-language-model-from-scratch).

## Project Overview

The project demonstrates all major steps of building, pretraining, and fine-tuning a transformer-based language model:
- **Stage 1:** Foundation model pretraining (unsupervised)
- **Stage 2:** Fine-tuning for classification (e.g., blog categorization)
- **Stage 3:** Instruction fine-tuning to build a personal assistant/chatbot

## Key Components

- `blog_classifier/` - Code and weights for a DistilBERT-based blog classifier (World, Sports, Business, Sci/Tech)
- `models_alpaca/` - Final instruction-tuned LLM weights and configs (using Alpaca for best results)
- `code_notebooks/` - Jupyter notebooks for training, inference, and pushing models to GitHub:
  - `project_run_all_model.ipynb` (Colab): Full pipeline runner
  - `blog_classifier.ipynb` (Colab): Blog classifier training
  - `blog_classifier_using.ipynb` (Colab): Blog classifier inference
  - `Pushing_blog_classifier_to_github.ipynb` (Colab): Git/GitHub automation
  - `Run all code` - This is the main code to use this project

## Features

- **Blog classification** with high accuracy
- **Instruction-following LLM** (Alpaca-tuned) for summarization, keyword extraction, and Q&A
- **Mindmap and main points extraction** from text
- **Easy-to-use Colab/Kaggle notebooks**

## Model Details

- **Final code uses an Alpaca instruction-tuned model** for higher quality results in instruction following and text generation tasks.
- All models are implemented in PyTorch and Hugging Face Transformers.

## Usage

1. Clone the repo and follow the notebook instructions in `code_notebooks/`.
2. For blog classification, use the scripts in `blog_classifier/`.
3. For advanced summarization and instruction tasks, use the Alpaca-based model in `models/`.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NLTK
- scikit-learn
- matplotlib, networkx (for mindmaps)

## Credits

Based on [Sebastian Raschka's book](https://www.manning.com/books/build-a-large-language-model-from-scratch) & inspired by the Stanford Alpaca project.

---

**Note:**  
The final code and demos use an Alpaca instruction-tuned model for best results in instruction-following and generation tasks.
