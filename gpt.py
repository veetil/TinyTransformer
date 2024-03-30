import config
import math 
import torch.nn as nn
from torch.nn import functional as F
import torch 


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


class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ff_layer1 = nn.Linear(config.EMBED, config.EMBED * config.FF_EXP, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.EMBED * config.FF_EXP, config.EMBED, bias=config.bias)
        self.dropout = nn.Dropout(p=config.FC_DROPOUT)

    def forward(self, x):
        hidden_out = self.gelu(self.ff_layer1(x))
        out = self.dropout(self.c_proj(hidden_out))
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        assert( config.EMBED % config.N_HEAD == 0 ) 
        self.c_attn = nn.Linear( config.EMBED , 3*config.EMBED, bias = config.bias )

        self.attn_dropout = nn.Dropout(config.ATTN_DROPOUT )
        self.c_proj = nn.Linear( config.EMBED , config.EMBED, bias = config.bias )
        self.residual_dropout = nn.Dropout(config.RESID_DROPOUT )
        self.register_buffer("bias", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                        .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))

    
    def forward(self,x):
        B, T, C = x.size()

        q, k, v =  self.c_attn(x).split(self.config.EMBED, dim = 2 )
        q = q.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2) # B, nh, T, C//nh 
        k = k.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2)
        v = v.view( B, T, self.config.N_HEAD, C // self.config.N_HEAD  ).transpose(1,2)

        att = q@k.transpose(-2,-1)/math.sqrt( k.size(-1) ) # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) 
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att,dim=-1)
        att = self.attn_dropout(att)
        y  = att@v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) 
        y = y.transpose(1,2).contiguous().view(B,T,C) #  (B, T, C) 

        y = self.c_proj(y)
        y = self.residual_dropout(y)
        return y 
    
 

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
        self.attn = SelfAttentionLayer(config)
        self.ln_2 = nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
        self.mlp =  FeedForwardLayer( config ) 

    def forward(self,x):
        x_ = self.ln_1(x)
        x = x +  self.attn(x_)
        x_ = self.ln_2(x)
        x = x + self.mlp(x_) 
        return x 

class GPT2Model( nn.Module ):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.VOCAB, config.EMBED)
        self.wpe = nn.Embedding(config.MAX_POS_EMBED, config.EMBED)
        self.drop = nn.Dropout( p =config.EMBED_DROPOUT )
        self.h = [ GPT2Block(config) for _ in range(config.LAYERS) ]

        self.ln_f = nn.LayerNorm(config.EMBED, eps=config.EPS, bias = config.bias)
        self.lm_head = nn.Linear( config.EMBED, config.VOCAB, bias = False )

        ## initial weight-tying, but how to make sure it stays tied 
        ## TODO: make sure the weights stay tied
        self.wte.weight  = self.lm_head.weight 

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
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

        x_t = self.wte(idx)
        x_p = self.wpe(pos)
        x = x_t + x_p
        x = self.drop(x)
        for h_ in self.h:
            x = h_(x)
        
        x = self.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy( logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            ## return only the last token in each batch
            x_ = x[:,[-1],:]
            logits = self.lm_head(x_) # (B,T,V)
            loss  = None

        return logits, loss 

