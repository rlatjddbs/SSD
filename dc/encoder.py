"""shout-out to https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, repeat, reduce
from collections import namedtuple


Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return x + emb[None, :, :]

    
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim,
                 dim_head=64,
                 heads=8,
                 mask=None,
                 dr=0.,
                 ):
        super().__init__()
        self.scale = (dim_head * heads) ** -0.5
        self.heads = heads
        self.mask = mask
        
        hidden_dim = dim_head * heads
        
        self.Wq = nn.Linear(dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dr)
        self.out = nn.Linear(hidden_dim, dim)
        
    def forward(self, 
                x, 
                context=None, 
                mask=None, 
                context_mask=None,
                sinusoidal_emb=None,
                prev_score=None,
                mem=None
                ):
        # assert exists(context) == exists(context_mask)
        
        b, h, *_ = x.shape
        device = x.device
        
        q_input = x
        k_input = default(context, x)
        v_input = default(context, x)
        
        q = self.Wq(q_input)
        k = self.Wk(k_input)
        v = self.Wv(v_input)
        
        q, k, v = map(lambda t: rearrange(t, 'bs horizon (heads dim) -> bs heads horizon dim', heads=self.heads), (q, k, v))
        
        # If there exists mask or context_mask:
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, h), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, h), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = torch.finfo(dots.dtype).max
        
        pre_softmax_attn = dots
        
        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask
        
        score = F.softmax(dots, dim=-1)
        post_softmax_attn = score
        score = self.dropout(score)
        
        out = einsum('b h i j, b h j d -> b h i d', score, v)
        out = rearrange(out, 'bs heads horizon dim -> bs horizon (heads dim)')
        
        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.out(out), intermediates  


class AttentionLayers(nn.Module):
    def __init__(self, 
                 dim,
                 depth,
                 heads=8,
                 cross_attend=True,
                 position_infused_attn=False,
                 pre_norm=True,
                 dr=0.,
                 ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.cross_attend = cross_attend
        self.layer_types = ('a', 'f')
        self.pre_norm = pre_norm
        self.dropout = nn.Dropout(dr)
        
        for _ in range(depth):    
            for layer_type in self.layer_types:
                if layer_type == 'a' or layer_type == 'c':
                    self.layers.append(nn.ModuleList([nn.LayerNorm(dim), MultiHeadAttention(dim, heads), Residual()]))
                elif layer_type == 'f':
                    self.layers.append(nn.ModuleList([nn.Identity(), FeedForward(dim), nn.Identity()]))
                else:
                    NotImplementedError(layer_type)
                    
    
    def forward(self, 
                x, 
                context=None, 
                mask=None, 
                context_mask=None, 
                return_hiddens=False
                ):
        assert exists(context) == self.cross_attend
        hiddens = []
        intermediates = []
        # prev_score = None
        # prev_cross_score = None
        
        mems = [None] * self.depth
        
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)
            
            # if layer_type == 'a':
            #     hiddens.append(x)
            #     layer_mem = mems.pop(0)

            residual = x
            x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask)
                out = self.dropout(out)
                x = residual_fn(out, residual)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask)
                out = self.dropout(out)
                x = residual_fn(out, residual)
            elif layer_type == 'f':
                out = block(x)


            # if layer_type in ('a', 'c'):
            #     intermediates.append(inter)

            # if layer_type == 'a': 
            #     prev_score = inter
            # elif layer_type == 'c': 
            #     prev_cross_score = inter

            # if not self.pre_norm and not is_last:
            #     x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x

    
class TransformerEmbedder(nn.Module):
    def __init__(self, dim, dim_attn, num_layer, dr=0.0, device='cuda'):
        super().__init__()
        self.device = device
        # self.embed_dim = embed_dim

        self.dropout = nn.Dropout(dr)
        self.project_x = nn.Linear(dim, dim_attn) if dim != dim_attn else nn.Identity()
        self.project_c = nn.Linear(dim, dim_attn) if dim != dim_attn else nn.Identity()
        self.attn_layers = AttentionLayers(dim=dim_attn, depth=num_layer, cross_attend=False, dr=dr)
        self.norm = nn.LayerNorm(dim_attn)
        
        
    def forward(self, x, context=None, mask=None, context_mask=None, return_attn=False):
        b = x.shape[0]
        
        x = x.to(self.device)
        
        x = rearrange(x, 'b t h -> b h t')
        x = self.dropout(x)
        x = self.project_x(x)
        if exists(context):
            context = rearrange(context, 'b t h -> b h t')
            context = self.dropout(context)
            context = self.project_c(context)
            
        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, context_mask=context_mask)
        z = self.norm(x)
        
        z = rearrange(z, 'b h t -> b t h')
        
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return z, attn_maps
        
        return z
    
