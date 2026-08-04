#!/bin/bash
set -e

echo "Installing Ollama..."

sudo apt-get update
sudo apt-get install -y curl
sudo apt-get install zstd

curl -fsSL https://ollama.com/install.sh | sh

echo "Verifying installation..."

ollama --version
