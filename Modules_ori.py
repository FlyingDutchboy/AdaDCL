import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

        
class NoiseTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 3:
            pass 
        else:
            raise ValueError(f"Expected input x with shape [B, D] or [B, 1, D], but got {x.shape}")

        out = self.transformer(x) 
        out = out.squeeze(1)      

        if residual:
            out = out + x.squeeze(1)

        return out


    
class NPNet(nn.Module):
    def __init__(self, num_items: int, dim: int, with_text_emb: bool = True):
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.with_text_emb = with_text_emb

        if self.with_text_emb:
            self.text_embedding = nn.Linear(dim, dim)

        self.noise_transformer = NoiseTransformer(d_model=dim, nhead=4, num_layers=2)

        self._alpha = nn.Parameter(torch.ones(1))  
        self._beta = nn.Parameter(torch.ones(1))

        self.modulation_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, dim * 6)
        )


        with torch.no_grad():
            self.modulation_head[-1].weight.fill_(0.0)
            bias_init = torch.zeros(dim * 6)

            # 设置 scale 相关的 bias 为 0.5413 (softplus(0.5413) ≈ 1.0)
            # 索引 0, 2, 4 是 scale
            bias_init[0 * dim: 1 * dim] = 0.5413  # scale_m
            bias_init[2 * dim: 3 * dim] = 0.5413  # scale_std
            bias_init[4 * dim: 5 * dim] = 0.5413  # scale_z


            # 索引 1, 3, 5 是 shift，保持为 0
            self.modulation_head[-1].bias.copy_(bias_init)

    def generate_modulation_params(self, guide_info):
        raw_params = self.modulation_head(guide_info)  # [Batch, Dim*6]
        chunks = torch.chunk(raw_params, 6, dim=-1)
        p_scale_m, p_shift_m, p_scale_std, p_shift_std, p_scale_z, p_shift_z = chunks
        scale_m = F.softplus(p_scale_m)
        shift_m = p_shift_m
        scale_std = F.softplus(p_scale_std) + 1e-4  # 防止变为0
        shift_std = p_shift_std
        scale_z = F.softplus(p_scale_z)
        shift_z = p_shift_z
        return scale_m, shift_m, scale_std, shift_std, scale_z, shift_z


    def forward(self, initial_noise: torch.Tensor, prompt_embeds: torch.Tensor = None) -> torch.Tensor:
        if self.with_text_emb and prompt_embeds is not None:
            encoder_hidden_states_vae = initial_noise                                  # 不加预训练嵌入，只是用vae噪声
            encoder_hidden_states_embedding = initial_noise
        else:
            encoder_hidden_states_vae = initial_noise
            encoder_hidden_states_embedding = initial_noise

        info_embedding = self.noise_transformer(encoder_hidden_states_embedding)       # 分支 B: 上下文重编码 (Transformer)
        vae_noise = encoder_hidden_states_vae                                          # 分支 A: 直接使用VAE生成的噪声

        info_noise = vae_noise + self._beta * info_embedding


        return info_noise
