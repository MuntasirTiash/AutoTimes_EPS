# models/AutoTimes_Llama.py
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from layers.mlp import MLP
from models.text_fusion import CrossAttentionBlock


def _str_to_torch_dtype(name: str) -> torch.dtype:
    name = (name or "float32").lower()
    if name in ("fp32", "float32"): return torch.float32
    if name in ("fp16", "float16"): return torch.float16
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32




class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # ---------- device ----------
        if getattr(configs, "use_multi_gpu", False):
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        # ---------- sizes ----------
        self.token_len   = int(getattr(configs, "token_len", 4))
        self.hidden_size = int(getattr(configs, "hidden_dim_of_gpt2", 4096))  # TS embed dim (keep arg name for compat)
        self.token_num   = int(getattr(configs, "token_num", 1))

        # ---------- TS tokenizer / detokenizer ----------
        self.encoder = MLP(
            self.token_len, self.hidden_size,
            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
            configs.dropout, configs.mlp_activation
        )
        self.decoder = MLP(
            self.hidden_size, self.token_len,
            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
            configs.dropout, configs.mlp_activation
        )

        # ---------- LLaMA backbone (local, offline) ----------
        llama_name = getattr(configs, "llama_model_name", "/ssd1/muntasir/Desktop/AutoTimes/llama-7b")
        llama_cfg  = AutoConfig.from_pretrained(llama_name, local_files_only=True)

        # If TS hidden != LLaMA hidden, add thin projections
        if self.hidden_size != llama_cfg.hidden_size:
            print(f"[WARN] hidden_size ({self.hidden_size}) != llama.hidden_size ({llama_cfg.hidden_size}); projecting.")
            self.ts2llama = nn.Linear(self.hidden_size, llama_cfg.hidden_size, bias=False)
            self.llama2ts = nn.Linear(llama_cfg.hidden_size, self.hidden_size, bias=False)
            backbone_dim = llama_cfg.hidden_size
        else:
            self.ts2llama = nn.Identity()
            self.llama2ts = nn.Identity()
            backbone_dim = self.hidden_size

        self.llama = AutoModel.from_pretrained(
            llama_name,
            torch_dtype=_str_to_torch_dtype(getattr(configs, "llama_dtype", "float32")),
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(self.device)

        if getattr(configs, "llama_grad_ckpt", False):
            self.llama.gradient_checkpointing_enable()

        # Freeze LLaMA by default
        self.freeze_llama = bool(getattr(configs, "freeze_llama", True))
        if self.freeze_llama:
            for p in self.llama.parameters():
                p.requires_grad = False

        # ---------- optional text fusion (10-Q) ----------
        self.use_text   = bool(getattr(configs, "use_text", False))
        self.text_mode  = getattr(configs, "text_mode", "emb")  # "emb" (precomputed) or "ids" (tokenized)
        if self.use_text:
            self.text_dim    = int(getattr(configs, "text_dim", backbone_dim))  # used only in "emb" mode
            self.cross_heads = int(getattr(configs, "cross_heads", 4))
            self.cross_ff    = int(getattr(configs, "cross_ff", 1024))
            self.cross = CrossAttentionBlock(
                d_model=backbone_dim,
                d_text=(self.text_dim if self.text_mode == "emb" else backbone_dim),
                n_heads=self.cross_heads,
                ff_hidden=self.cross_ff,
                p_drop=getattr(configs, "dropout", 0.1),
            )

        # (optional) your mark mixing behavior
        self.mix = bool(getattr(configs, "mix", False))
        self.add_scale = float(getattr(configs, "add_scale", 1.0))

    # -------- helpers --------
    def _norm(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev

    def _llama_forward(self, **kwargs):
        """
        Wrap LLaMA forward to avoid grad graph when frozen (saves VRAM).
        """
        if self.freeze_llama:
            with torch.no_grad():
                out = self.llama(**kwargs).last_hidden_state
        else:
            out = self.llama(**kwargs).last_hidden_state
        return out

    # -------- main path --------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, C]  (C variables; last channel is target)
        x_enc, means, stdev = self._norm(x_enc)

        B, T, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(B * C, -1)  # [B*C, T]

        # split into non-overlapping tokens of length token_len
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)  # [B*C, token_num, token_len]
        token_num = fold_out.shape[1]

        # TS tokenizer -> embeddings
        ts_emb = self.encoder(fold_out)  # [B*C, token_num, hidden_size]
        if not isinstance(self.ts2llama, nn.Identity):
            ts_emb = self.ts2llama(ts_emb)  # [B*C, token_num, llama_hidden]

        # Optional mark mixing (expects broadcastable shape)
        if self.mix and x_mark_enc is not None:
            ts_emb = ts_emb / (ts_emb.norm(dim=2, keepdim=True) + 1e-8)
            x_mark = x_mark_enc / (x_mark_enc.norm(dim=2, keepdim=True) + 1e-8)
            ts_emb = ts_emb + self.add_scale * x_mark

        # ===== Text cross-attention (optional) =====
        if self.use_text and x_dec is not None:
            if isinstance(x_dec, dict) and "input_ids" in x_dec:
                # "ids" mode: run LLaMA on text tokens to get contextual embeddings
                text_inputs = {k: v.to(self.device) for k, v in x_dec.items()}   # [B, T_txt]
                text_hidden = self._llama_forward(**text_inputs)                  # [B, T_txt, llama_hidden]
                text_hidden = text_hidden.repeat_interleave(C, dim=0)            # [B*C, T_txt, llama_hidden]
                txt_for_cross = text_hidden
            else:
                # "emb" mode: precomputed embeddings [B, T_txt, D_text]
                if x_dec.dim() == 2:
                    x_dec = x_dec.unsqueeze(1)  # [B, 1, D]
                x_dec = x_dec.to(self.device).repeat_interleave(C, dim=0)  # [B*C, T_txt, D_text]
                txt_for_cross = x_dec

            ts_emb = self.cross(ts_emb, txt_for_cross)  # [B*C, token_num, llama_hidden]

        # LLaMA over TS token embeddings
        attn_mask = torch.ones((ts_emb.shape[0], ts_emb.shape[1]), dtype=torch.long, device=ts_emb.device)
        llama_out = self._llama_forward(inputs_embeds=ts_emb, attention_mask=attn_mask)  # [B*C, token_num, llama_hidden]

        # map back to TS hidden if we projected
        if not isinstance(self.llama2ts, nn.Identity):
            llama_out = self.llama2ts(llama_out)  # [B*C, token_num, hidden_size]

        # detokenize back to time
        dec_out = self.decoder(llama_out)                 # [B*C, token_num, token_len]
        dec_out = dec_out.reshape(B, C, -1).permute(0, 2, 1)  # [B, token_num*token_len, C]

        # de-normalize
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
