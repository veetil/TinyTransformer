# GPT-2 Implementation from Scratch

Welcome to the GitHub repository of my from-scratch implementation of GPT-2. My goal, in addion to building  GPT-2 from the ground up, is to study the architecture, impacts of other architectures and scaling laws. The initiative serves as an educational resource for myself and others interested in understanding and experimenting with the mechanisms of GPT-2. 

## Project Status


<p align="center">
  <img src="images/val-loss-gpt2-swiglu-rmsnorm.jpg" alt="Validatoin loss GPT-2 with SwiGLU and RMS Norm" width="800">
</p>

Here we see the validation loss for the original GPT-2 implementation vs a customized version, with SwiGLU and RMS Norm. The regular Feedforward layer with 2 linear layers is replaced with a Feed Forward Network based on SwiGLU ( https://arxiv.org/pdf/2002.05202.pdf%5C)%E5%BC%95%E5%85%A5%E7%9A%84 ). The RMS norm replaces the regular normalization layer. 
This leads to an improvement in training. Although the inmprovement is not obvious, the validatoin loss for 50k iterations is reached with 30k iterations using the modified architecture. 
The FFN SwiGLU parameters are adjusted so that its total parameter count approximately matches the regular FFN for a more fair comparison. 
Caveats
- Hyperparameter optimization is required to fully establish the differences. Here, we note the trends at a single setting of all hyperparams
- Droput was used after the output of regular FFN, but not in SwiGLU FFN
- Scaled initilization was used to initialize weights of the final layer of regular FFN (per GPT-2 paper), but not in SwiGLU FFN. Instead, the default mean 0 sigma 0.02 initialization was used. 


## Features

- **Replication of GPT-2 Architecture**: Implements the multi-layer transformer model, for the most part, as described in the original GPT-2 paper. OpenAI tensorflow implementation, Pytorch GPT-2 implementation and Karpathy's nanoGPT project have been important references in this atttempt


## Training
This assumes that you have a cluster of GPUs, similar to vast.ai, with CUDA, pytorch installed. 
First, install dependencies. Prepare data. Then launch DDP run, assunming single node with 8 GPUs. 
OpenWebText dataset is downloaded by default. 
git clone https://github.com/veetil/gpt
cd gpt 
pip install -r requirements.txt
python data/prepare.py 
wandb login
torchrun --standalone --nproc_per_node=8 train.py