from typing import Optional

import torch
from torch import nn, Tensor

from module.Attention import Attention
from utils.AttnBlocksConf import AttnBlocksConf

class MHABlock(nn.Module):
    def __init__(self, 
            attnConf: AttnBlocksConf = None,
            dropout: float = 0.1,
            intermediateDim: int = None,
            batch_first: bool = True,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32
    )->None:
        super(MHABlock, self).__init__()

        hidDim = attnConf.hidDim
        
        self._mha = Attention(
            attnConf=attnConf,
            device=device,
            dtype=dtype
        )

        self.norm = nn.LayerNorm(hidDim, device=device, dtype=dtype)

        intermediateDim = intermediateDim if intermediateDim is not None else 4*hidDim
        self.ffn = nn.Sequential(
            nn.Linear(hidDim, intermediateDim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(intermediateDim, hidDim, device=device, dtype=dtype),
            nn.Dropout(dropout)
        )
        self.outNorm = nn.LayerNorm(hidDim, device=device, dtype=dtype)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False
        )->Tensor:

        residual = query

        query, attnWeight = self._mha(
            query=query,
            kv=key,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )
        
        query = self.norm(query + residual)

        residual = query
        query = self.ffn(query)
        query = self.outNorm(query + residual)

        return query, attnWeight