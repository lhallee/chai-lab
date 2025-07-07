#!/bin/bash

# chmod +x setup_chai.sh
# ./setup_chai.sh

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Python virtual environment for Chai-1..."

# Create virtual environment
python3 -m venv ~/chai_venv

# Activate virtual environment
source ~/chai_venv/bin/activate

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install torch and torchvision
echo "Installing torch and torchvision..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U

# Install requirements with force reinstall
echo "Installing requirements"
pip install -r requirements.txt

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source ~/chai_venv/bin/activate"
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: ~/chai_venv"
echo "======================="

