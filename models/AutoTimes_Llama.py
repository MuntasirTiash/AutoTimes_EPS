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
        # This is the original AutoTimes encoder/decoder, which treats each channel independently.
        # self.encoder = MLP(
        #     self.token_len, self.hidden_size,
        #     configs.mlp_hidden_dim, configs.mlp_hidden_layers,
        #     configs.dropout, configs.mlp_activation
        # )
        # self.decoder = MLP(
        #     self.hidden_size, self.token_len,
        #     configs.mlp_hidden_dim, configs.mlp_hidden_layers,
        #     configs.dropout, configs.mlp_activation
        # )
        # New encoder/decoder to process all covariates together
        self.encoder = None  # Will be nn.Linear(C, hidden_size)
        self.decoder = None  # Will be nn.Linear(hidden_size, 1) for target prediction

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

        # Lazy initialization of encoder/decoder
        if self.encoder is None:
            C = x_enc.shape[-1]
            self.encoder = nn.Linear(C, self.hidden_size).to(self.device)
            self.decoder = nn.Linear(self.hidden_size, 1).to(self.device)

        # 1. Normalize the input covariates.
        # The normalization is instance-wise, which means it's done on a per-sample basis.
        # This helps the model focus on the patterns of the time series rather than the scale.
        x_enc, means, stdev = self._norm(x_enc)

        B, T, C = x_enc.shape

        # 2. Encode the covariates.
        # Instead of breaking down the time series into tokens and processing each channel independently,
        # we now project the C covariates at each time step into the hidden dimension.
        # This allows the model to learn the relationships between the covariates at each time step.
        # Input: [B, T, C] -> Output: [B, T, hidden_size]
        ts_emb = self.encoder(x_enc)
        if not isinstance(self.ts2llama, nn.Identity):
            ts_emb = self.ts2llama(ts_emb)  # [B, T, llama_hidden]

        # Optional mark mixing (expects broadcastable shape)
        if self.mix and x_mark_enc is not None:
             raise NotImplementedError("Mark mixing not implemented for this architecture yet")


        # 3. Text cross-attention (optional).
        # This part remains the same, but we no longer need to repeat the text embeddings for each channel.
        if self.use_text and x_dec is not None:
            if isinstance(x_dec, dict) and "input_ids" in x_dec:
                # "ids" mode: run LLaMA on text tokens to get contextual embeddings
                text_inputs = {k: v.to(self.device) for k, v in x_dec.items()}   # [B, T_txt]
                text_hidden = self._llama_forward(**text_inputs)                  # [B, T_txt, llama_hidden]
                txt_for_cross = text_hidden
            else:
                # "emb" mode: precomputed embeddings [B, T_txt, D_text]
                if x_dec.dim() == 2:
                    x_dec = x_dec.unsqueeze(1)  # [B, 1, D]
                x_dec = x_dec.to(self.device)
                txt_for_cross = x_dec

            ts_emb = self.cross(ts_emb, txt_for_cross)  # [B, T, llama_hidden]

        # 4. Pass the embeddings through the LLaMA backbone.
        # The attention mask ensures that the model attends to all the tokens.
        # Input: [B, T, llama_hidden] -> Output: [B, T, llama_hidden]
        attn_mask = torch.ones((ts_emb.shape[0], ts_emb.shape[1]), dtype=torch.long, device=ts_emb.device)
        llama_out = self._llama_forward(inputs_embeds=ts_emb, attention_mask=attn_mask)

        # map back to TS hidden if we projected
        if not isinstance(self.llama2ts, nn.Identity):
            llama_out = self.llama2ts(llama_out)  # [B, T, hidden_size]

        # 5. Decode the LLaMA output to get the target prediction.
        # The decoder projects the hidden state at each time step to a single value, which is the prediction for the target variable.
        # Input: [B, T, hidden_size] -> Output: [B, T, 1]
        dec_out = self.decoder(llama_out)

        # 6. De-normalize the output.
        # We use the mean and standard deviation of the target variable (the last channel) to de-normalize the output.
        target_means = means[:, :, -1:]
        target_stdev = stdev[:, :, -1:]
        dec_out = dec_out * target_stdev + target_means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
