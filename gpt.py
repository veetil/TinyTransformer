import config
import math 
import torch.nn as nn
from torch.nn import functional as F
import torch 
from collections import namedtuple
import inspect 
from typing import Any, Optional, Tuple
from typing import List
import torch.distributed as dist
import os 
from torch import Tensor
## This code implements the GPT-2 model as described in the paper "Language Models are Unsupervised Multitask Learners" by Radford et al.
## Author: Vineeth Veetil 
## The code is based on the OpenAI GPT-2 implementation, but has been modified to be more readable and to allow for easy modification.
## Below is the GPT-2 architecture as implemented in this code.
## GPT architecture
"""
GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)            
"""

## Reference https://arxiv.org/pdf/2204.02311.pdf
class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.EMBED, config.EMBED * config.FF_EXP, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.EMBED * config.FF_EXP, config.EMBED, bias=config.bias)
        self.dropout = nn.Dropout(p=config.FC_DROPOUT)

    def forward(self, x):
        hidden_out = self.gelu(self.c_fc(x))
        out = self.dropout(self.c_proj(hidden_out))
        return out

class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        config 
    ):
        super().__init__()
        dim = config.EMBED
        hidden_dim = config.EMBED * config.FF_EXP
        ffn_dim_multiplier = None 
        multiple_of = 1 

        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        ## TODO fix initialization method 
        ## FFNSwiGLU(x, W, V, W2) = (Swish1(xW) ⊗ xV )W2
        self.v = nn.Linear( dim, hidden_dim, bias = False ) 
        self.w = nn.Linear( dim, hidden_dim, bias = False ) 

##      Use the commented version ( also in forward() ) if you dont want scaled init applied to SwiGLU final layer weights
        self.w2     = nn.Linear( hidden_dim, dim, bias = False ) 
##        self.c_proj = nn.Linear( hidden_dim, dim, bias = False ) 

        ## Add Dropout to match regular FFN implementation
        ## self.dropout = nn.Dropout(p=config.FC_DROPOUT)


    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))
#        return self.c_proj(F.silu(self.w(x)) * self.v(x))
    
#       nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
#    print('freqs cis shape',freqs_cis.shape)
#    print('x shape',x.shape)
    
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i == ndim-2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
def apply_rotary_emb(
    xq: Tensor,
    xk: Tensor,
    freqs_cis: Tensor,
) -> Tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, config ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.config = config

    def forward(self, inputs: Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.config.TOPK_EXPERTS)
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            (batch_idx, token_idx, nth_expert) = torch.where(selected_experts == i)
            results[batch_idx,token_idx] += weights[batch_idx, token_idx, nth_expert, None] * expert(inputs[batch_idx, token_idx])
            
        return results


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:  # type: ignore
        ctx.group =  dist.group.WORLD
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=ctx.group)
        return output
    
def one_hot(tensor: Tensor, num_classes: int) -> Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def gating(logits: torch.Tensor) : 
    # NOTE(msb) softmax requires FP32: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
    gates = F.softmax(logits, dim=1, dtype=torch.float)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]

    capacity =  num_tokens 
    assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    gates1_s = (gates * mask1).sum(dim=1)  # einsum("se,se->s")
    gates2_s = (gates * mask2).sum(dim=1)  # einsum("se,se->s")
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return combine_weights.to(logits.dtype), dispatch_mask


class MoeLayer_ddp(nn.Module):
    def __init__(self, experts: nn.Module, gate: nn.Module, config, scan_expert_func=None):
        super().__init__()
        
#        assert len(experts) > 0
        self.experts = experts
        self.gate = gate
        self.config = config
        self.world_size = dist.get_world_size()
        self.num_local_experts = 1 #len(experts) // self.world_size

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)


    def forward(self, inputs: Tensor):

        d_model = inputs.shape[2]
        reshaped_input = inputs.reshape(-1, d_model) # (s , m)
        logits = self.gate(reshaped_input) # (s, e)
        combine_weights, dispatch_mask = gating(logits) # (s, e, c), (s, e, c)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input) # (e, c, m)
        dispatched_input = _AllToAll.apply( dispatched_input) 
        dispatched_input = dispatched_input.reshape(self.world_size, -1, d_model) # (g, c, m)
        expert = self.experts # [dist.get_rank()] # only local expert
        chunk = dispatched_input # (g, c, m)
        expert_output = expert(chunk) # (g, c, m)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size , -1, d_model) # (e, c, m)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(inputs.shape)





class SelfAttentionLayer(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        assert( config.EMBED % config.N_HEAD == 0 ) 
        self.c_attn = nn.Linear( config.EMBED , 3*config.EMBED, bias = config.bias )

        self.attn_dropout = nn.Dropout(config.ATTN_DROPOUT )
        self.c_proj = nn.Linear( config.EMBED , config.EMBED, bias = config.bias )
        self.resid_dropout = nn.Dropout(config.RESID_DROPOUT )
        # flash attention 
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
           self.register_buffer("bias", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                        .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))

    
    def forward(self,x,freqs_cis = None):
        B, T, C = x.size()

        q, k, v =  self.c_attn(x).split(self.config.EMBED, dim = 2 )
        q = q.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2) # B, nh, T, C//nh 
        k = k.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2)
        v = v.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2)

        if self.config.ROTARY_EMBED == 1 : 
#            print("applying rotary embedding in attention layer")
            q,k =  apply_rotary_emb(q, k, freqs_cis=freqs_cis)

#        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
#            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.DROPOUT , is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side


#        att = q@k.transpose(-2,-1)/math.sqrt( k.size(-1) ) # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) 
#        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
#        att = F.softmax(att,dim=-1)
#        att = self.attn_dropout(att)
#        y  = att@v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) 
#        y = y.transpose(1,2).contiguous().view(B,T,C) #  (B, T, C) 

        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y 
    
 


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        if config.RMS_NORM == 0 : 
            self.ln_1 = nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
        else:
            print("Instantiating RMSNorm 1")
            self.ln_1 = RMSNorm(config.EMBED, eps=config.EPS)

        self.attn = SelfAttentionLayer(config)

        if config.RMS_NORM == 0 : 
            self.ln_2 = nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
        else:
            print("Instantiating RMSNorm 2")
            self.ln_2 = RMSNorm(config.EMBED, eps=config.EPS)

        ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?


        if config.MoE == 1 : 
            if not ddp:
                self.mlp = MoeLayer(
                    experts=[FeedForwardSwiGLU(config) for _ in range(config.NUM_EXPERTS)],
                    gate=nn.Linear(config.EMBED, config.NUM_EXPERTS, bias=False),
                    config = config
                )
            else:

                self.mlp = MoeLayer_ddp(
                    #experts=[FeedForwardSwiGLU(config) for _ in range(config.NUM_EXPERTS)],
                    ## TODO: Fix this hacky way of instantiating experts. This assumes one device per expert
                    experts=FeedForwardSwiGLU(config) , 
                    gate=nn.Linear(config.EMBED, config.NUM_EXPERTS, bias=False),
                    config = config,
                    scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True)
                )
        elif config.SWIGLU == 1 : 
            print("Instantiating SwiGLU")
            self.mlp = FeedForwardSwiGLU( config )
        else:
            print("Instantiating regular FFN")
            self.mlp =  FeedForwardLayer( config ) 


    def forward(self,x,freqs_cis = None):
        x_ = self.ln_1(x)
        if self.config.ROTARY_EMBED == 0 : 
            x = x +  self.attn(x_)
        else:
            x = x +  self.attn(x_,freqs_cis)

        x_ = self.ln_2(x)
        x = x + self.mlp(x_) 
        return x 

class GPT2Model( nn.Module ):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ## Learned positional encoding
        if config.ROTARY_EMBED == 0 : 
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.VOCAB, config.EMBED),
                wpe = nn.Embedding(config.MAX_POS_EMBED, config.EMBED),
                drop = nn.Dropout(config.EMBED_DROPOUT),
                h = nn.ModuleList([GPT2Block(config) for _ in range(config.LAYERS)]),
                ln_f = nn.LayerNorm(config.EMBED, bias=config.bias),
            ))
        ## RoPE => no positional encoding at this stage , but precompute RoPE multipliers 
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.VOCAB, config.EMBED),
                drop = nn.Dropout(config.EMBED_DROPOUT),
                h = nn.ModuleList([GPT2Block(config) for _ in range(config.LAYERS)]),
                ln_f = nn.LayerNorm(config.EMBED, bias=config.bias),
            ))
            self.freqs_cis = precompute_freqs_cis( config.EMBED // config.N_HEAD, config.BLOCK_SIZE * 2)

        self.lm_head = nn.Linear( config.EMBED, config.VOCAB, bias = False )

        ## initial weight-tying, but how to make sure it stays tied 
        ## TODO: make sure the weights stay tied
        self.transformer.wte.weight  = self.lm_head.weight 

        self.apply(self._init_weights)


        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.LAYERS))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self,idx,targets=None):
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.BLOCK_SIZE, f"Cannot forward sequence of length {t}, block size is only {self.config.BLOCK_SIZE}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        x_t = self.transformer.wte(idx)

        if self.config.ROTARY_EMBED == 0 :
            x_p = self.transformer.wpe(pos)
            x = x_t + x_p
        else:
            x = x_t ## In RoPE, we rotate the query and key vectors, not input embedding directly 
            self.freqs_cis = self.freqs_cis.to(device)
            freqs_cis = self.freqs_cis[:self.config.BLOCK_SIZE]

        ## TODO - Clean up hack approach with variables inside conditions. 
        x = self.transformer.drop(x)
        for h_ in self.transformer.h:
            if self.config.ROTARY_EMBED == 0 : 
                x = h_(x)
            else:
                x = h_(x,freqs_cis)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy( logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            ## return only the last token in each batch
            x_ = x[:,[-1],:]
            logits = self.lm_head(x_) # (B,1,V)
            loss  = None

        return logits, loss 


    # Important setting 1: skip handling expert parameters by Pytorch DDP
    def add_param_to_skip_allreduce(self, param_name):
        if not hasattr(self, '_ddp_params_and_buffers_to_ignore'):
          self._ddp_params_and_buffers_to_ignore = list()
        self._ddp_params_and_buffers_to_ignore.append(param_name)


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(LAYERS=12, N_HEAD=12, EMBED=768),  # 124M params
            'gpt2-medium':  dict(LAYERS=24, N_HEAD=16, EMBED=1024), # 350M params
            'gpt2-large':   dict(LAYERS=36, N_HEAD=20, EMBED=1280), # 774M params
            'gpt2-xl':      dict(LAYERS=48, N_HEAD=25, EMBED=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['VOCAB'] = 50257 # always 50257 for GPT model checkpoints
        config_args['BLOCK_SIZE'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config_ = config_args
        config2 = config.read_config()
        ## for keys in config2, but not in config_, set the value of config2 to config_

        # Convert to OrderedDict
        config2_dict = config2._asdict()
        for k in config_.keys():
            config2_dict[k] = config_[k]
        MyDict = namedtuple('MyDict', config2_dict.keys())
        config2 = MyDict(**config2_dict)


        model = GPT2Model(config2)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        print('sd_keys_hf:', sd_keys_hf)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def get_num_params(self, config_ = None, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if  config_ is None or config_.ROTARY_EMBED == 0 : 
                n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.BLOCK_SIZE else idx[:, -self.config.BLOCK_SIZE:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
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


    def estimate_mfu(self, fwdbwd_per_iter, dt,config_ = None):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params(config_)
        cfg = self.config
        L, H, Q, T = cfg.LAYERS , cfg.N_HEAD, cfg.EMBED//cfg.N_HEAD, cfg.BLOCK_SIZE
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu