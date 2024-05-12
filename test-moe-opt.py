import torch.nn as nn
import torch
import config
from gpt import FeedForwardEye
from gpt import FeedForwardSwiGLU
from gpt import MoeLayer_ddp
import torch.distributed as dist
import os 
import inspect 
from torch.nn import functional as F
from contextlib import nullcontext
from torch.distributed import init_process_group, destroy_process_group

DEBUG = True
SKIP_GRAD_ALLREDUCE_EXPERTS = True

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
# Important setting 1: skip handling expert parameters by Pytorch DDP
def add_param_to_skip_allreduce(model, param_name):
    if not hasattr(model, '_ddp_params_and_buffers_to_ignore'):
        model._ddp_params_and_buffers_to_ignore = list()
    model._ddp_params_and_buffers_to_ignore.append(param_name)

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
                    experts=FeedForwardEye(config_) , 
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
    if  SKIP_GRAD_ALLREDUCE_EXPERTS : 
        for name, param in mlp.named_parameters():
            if hasattr(param, 'skip_allreduce'):
                add_param_to_skip_allreduce(mlp,name)

    if torch.cuda.is_available():
        new_val = local_rank*1.0*torch.eye(config_.EMBED, config_.EMBED)
        mlp.experts.c_fc.weight.data = new_val.to(device)

        print("local rank",local_rank,"wt",mlp.experts.c_fc.weight[0][0])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    optimizer =  configure_optimizers(mlp,weight_decay, learning_rate, (beta1, beta2), device_type)

    # Wrap the model with DDP based on the device
    if device_type == "cuda":
        mlp = nn.parallel.DistributedDataParallel(mlp, device_ids=[local_rank])
    else:
        mlp = nn.parallel.DistributedDataParallel(mlp)

    # Modify expert weights after DDP wrap
    if DEBUG : 
        if torch.cuda.is_available():
            new_val = local_rank * 1.0 * torch.eye(config_.EMBED, config_.EMBED)
            mlp.module.experts.c_fc.weight.data = new_val.to(device)
        
    input1 = input1.to(device)
    output = mlp(input1)

    # Backward pass
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    logits = output
    num1 = logits.view(-1, logits.size(-1)).shape[0]
    print(f"num1: {num1}")
    targets = torch.randint(0, logits.size(-1), (num1,), device=device)
    targets = targets.contiguous()
    ## initialize targets with 0 to embed
    print(f"targets: {targets}")
    loss = F.cross_entropy( logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    optimizer.zero_grad()
    scaler.scale(loss).backward()

    ## print the grad
#    for name, param in mlp.named_parameters():
#        print(f"Rank {local_rank}: {name}, grad: {param.grad}")


    master_process = ddp_local_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_local_rank # each process gets a different seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    # training loop
    local_iter_num = 0 # number of iterations in the lifetime of this process
#    raw_model = model.module if ddp else model # unwrap DDP container if needed
    lr = 1e-4 
    grad_clip = 1.0
    max_iters = 1000

    # define big random x,y and supply in each iteratio
    iter_num = 0
    X_set = torch.rand(10000, config_.BLOCK_SIZE, config_.EMBED)
    #Y_set is element wise cube
    Y_set = torch.pow(X_set, 3)
    X, Y = X_set[iter_num % X_set.size(0)], Y_set[iter_num % Y_set.size(0)]


    while True:

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # in DDP training we only need to sync gradients at the last micro step.
        # the official way to do this is with model.no_sync() context manager, but
        # I really dislike that this bloats the code and forces us to repeat code
        # looking at the source of that context manager, it just toggles this variable
        mlp.require_backward_grad_sync = True
        with ctx:
            logits, loss = mlp(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = X_set[(iter_num+1) % X_set.size(0)], Y_set[(iter_num+1) % Y_set.size(0)]
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1

        if iter_num % 10 == 0:
            print(f"Rank {local_rank}, iter {iter_num}, loss {loss.item()}")
            
        # termination conditions
        if iter_num > max_iters:
            break

    destroy_process_group()




if __name__ == "__main__":
    main()