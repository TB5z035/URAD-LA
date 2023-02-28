#! /usr/bin/env bash

set -ex

# Create a conda environment
conda create -yn lbsl python=3.8

# Install numpy and torch
conda install -yn lbsl numpy==1.22.3
conda install -yn lbsl pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

# Install the rest of the packages
conda activate lbsl
pip install -r requirements.txt