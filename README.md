# GPT-2 Implementation from Scratch

Welcome to the GitHub repository of my from-scratch implementation of GPT-2. My goal, in addion to building  GPT-2 from the ground up, is to study the architecture, impacts of other architectures and scaling laws. The initiative serves as an educational resource for myself and others interested in understanding and experimenting with the mechanisms of GPT-2. 

## Project Status

As of now, the first complete version of the GPT-2 model has been implemented and is ready for testing. 

## Features

- **Replication of GPT-2 Architecture**: Implements the multi-layer transformer model, for the most part, as described in the original GPT-2 paper. OpenAI tensorflow implementation, Pytorch GPT-2 implementation and Karpathy's nanoGPT project have been important references in this atttempt


## Training
This assumes that you have a cluster of GPUs, similar to vast.ai, with CUDA, pytorch installed. 
First, install dependencies. Prepare data. Then launch DDP run, assunming single node with 8 GPUs. 
The data is 
git clone https://github.com/veetil/gpt
cd gpt 
pip install -r requirements.txt
python data/prepare.py 
torchrun --standalone --nproc_per_node=8 train.py

