#!/bin/bash

# Modal Setup Script for Slug Search Training
echo "🚀 Setting up Modal for Slug Search Training..."

# Install Modal if not already installed
echo "📦 Installing Modal..."
uv pip install modal

# Setup Modal authentication
echo "🔐 Setting up Modal authentication..."
modal setup

# Create secrets for environment variables
echo "🔑 Creating Modal secrets..."

# W&B secret (optional)
read -p "Do you want to set up W&B logging? (y/n): " setup_wandb
if [[ $setup_wandb == "y" || $setup_wandb == "Y" ]]; then
    read -p "Enter your W&B API key: " wandb_key
    modal secret create wandb WANDB_API_KEY_NO_ART="$wandb_key"
    echo "✅ W&B secret created"
fi

# HuggingFace secret (optional)
read -p "Do you want to set up HuggingFace access? (y/n): " setup_hf
if [[ $setup_hf == "y" || $setup_hf == "Y" ]]; then
    read -p "Enter your HuggingFace token: " hf_token
    modal secret create huggingface HF_TOKEN="$hf_token"
    echo "✅ HuggingFace secret created"
fi

# OpenAI secret (optional)
read -p "Do you want to set up OpenAI access? (y/n): " setup_openai
if [[ $setup_openai == "y" || $setup_openai == "Y" ]]; then
    read -p "Enter your OpenAI API key: " openai_key
    modal secret create openai OPENAI_API_KEY="$openai_key"
    echo "✅ OpenAI secret created"
fi

# Embedder secret (for your embedder service)
read -p "Do you want to set up Embedder API access? (y/n): " setup_embedder
if [[ $setup_embedder == "y" || $setup_embedder == "Y" ]]; then
    read -p "Enter your Embedder API key: " embedder_key
    modal secret create embedder EMBEDDER_API_KEY="$embedder_key"
    echo "✅ Embedder secret created"
fi

echo ""
echo "🎉 Modal setup complete!"
echo ""
echo "📋 Quick usage commands:"
echo ""
echo "  🎯 Debug training with embedder (RECOMMENDED - matches your local setup):"
echo "    modal run train_modal.py::debug"
echo ""
echo "  🎯 Debug training without embedder (will likely fail):"
echo "    modal run train_modal.py::debug_no_embedder"
echo ""
echo "  🚀 Production training:"
echo "    modal run train_modal.py::production"
echo ""
echo "  ⚙️  Custom training:"
echo "    modal run train_modal.py::custom --debug=true --chat_template=\"qwen3_preserve_thinking\""
echo ""
echo "💡 All training logs and model artifacts will be persisted in Modal volumes"
echo "   and accessible across runs."
echo ""
echo "🔧 GPU Configuration:"
echo "   Currently set to H100. Edit train_modal.py to change GPU type:"
echo "   - modal.gpu.A100(count=1)"
echo "   - modal.gpu.T4(count=1)"
echo "   - modal.gpu.H100(count=1)"
echo ""
echo "🔍 The debug function includes:"
echo "   - Embedder vLLM server (BAAI/bge-large-en-v1.5) on localhost:40002"
echo "   - Your exact training command with --debug --chat_template qwen3_preserve_thinking"
echo "   - Real-time output streaming and log file creation (debug80.log)" 