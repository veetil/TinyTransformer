import torch.nn as nn
import torch
import config
from gpt import FeedForwardSwiGLU
from gpt import MoeLayer_ddp
import torch.distributed as dist
import os
import inspect 
from torch.nn import functional as F

def configure_optimizers(self_, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self_.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    device = 'cpu'
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if ddp : 
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

    config_ = config.read_config()
    config.print_config(config_)
    torch.manual_seed(42)

    # Generate input tensor
    input1 = torch.rand(8, config_.BLOCK_SIZE, config_.EMBED)

    # Create the model
    mlp = MoeLayer_ddp(
                    experts=FeedForwardSwiGLU(config_) , 
                    gate=nn.Linear(config_.EMBED, config_.NUM_EXPERTS, bias=False),
                    config = config_,
                    scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True)
                )

    if ddp : 
        if torch.cuda.is_available():
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    weight_decay = 1e-1
    learning_rate = 6e-4 # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95


    # Move the model and input to the device
    mlp.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


    optimizer =  configure_optimizers(mlp,weight_decay, learning_rate, (beta1, beta2), device_type)

    # Wrap the model with DDP based on the device
    if device_type == "cuda":
        mlp = nn.parallel.DistributedDataParallel(mlp, device_ids=[local_rank])
    else:
        mlp = nn.parallel.DistributedDataParallel(mlp)

    input1 = input1.to(device)

    # Forward pass
    output = mlp(input1)
    print(f"Rank {local_rank}: Input shape: {input1.shape}, Output shape: {output.shape}")
    if local_rank == 0:
        print(f"Rank {local_rank}: Output: {output}")
    print(f"Rank {local_rank}: Output: {output[0, 0, :5]}")

    # Backward pass
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    logits = output
    num1 = logits.view(-1, logits.size(-1)).shape[0]
    print(f"num1: {num1}")
    targets = torch.randint(0, logits.size(-1), (num1,), device=device)
    ## initialize targets with 0 to embed
    print(f"targets: {targets}")
    loss = F.cross_entropy( logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    optimizer.zero_grad()
    scaler.scale(loss).backward()

    dist.destroy_process_group()






    while   False:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            loss = 1.0 
            optimizer.zero_grad()
            scaler.scale(loss).backward()






if __name__ == "__main__":
    main()