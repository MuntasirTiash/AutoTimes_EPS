# models/text_fusion.py
import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """Fuse time-series token embeddings (Q) with text embeddings (K,V).

    Shapes:
      ts_tokens:  [B*, T_ts, d_model]
      text_tokens:[B*, T_txt, d_text]  (T_txt can be 1 if you use a pooled doc embedding)
    Returns:
      fused:      [B*, T_ts, d_model]

    Args:
        d_model (int): The dimension of the time-series token embeddings.
        d_text (int): The dimension of the text embeddings.
        n_heads (int, optional): The number of attention heads.
            Defaults to 4.
        ff_hidden (int, optional): The number of hidden units in the
            feed-forward network. Defaults to 1024.
        p_drop (float, optional): The dropout probability. Defaults to 0.1.
    """
    def __init__(self, d_model: int, d_text: int, n_heads: int = 4, ff_hidden: int = 1024, p_drop: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(d_text, d_model, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(ff_hidden, d_model),
        )
        self.ff_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, ts_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        kv = self.text_proj(text_tokens)              # [B*, T_txt, d_model]
        # Cross-attend: Q=ts, K=V=projected text
        attn_out, _ = self.attn(ts_tokens, kv, kv)    # [B*, T_ts, d_model]
        ts_tokens = self.attn_ln(ts_tokens + self.drop(attn_out))
        ff_out = self.ff(ts_tokens)
        fused = self.ff_ln(ts_tokens + self.drop(ff_out))
        return fused