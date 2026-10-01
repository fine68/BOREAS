from __future__ import annotations

import torch
from torch import nn


STATE_TEMPS_LO, STATE_TEMPS_HI = 0, 14
STATE_HUMID_LO, STATE_HUMID_HI = 14, 28
STATE_LAT_LO, STATE_LAT_HI = 28, 37
STATE_EAT_LO, STATE_EAT_HI = 37, 46
STATE_ONOFF_LO, STATE_ONOFF_HI = 46, 55
STATE_FAN_LO, STATE_FAN_HI = 55, 64
STATE_VALVE_LO, STATE_VALVE_HI = 64, 73
STATE_U_LO, STATE_U_HI = 46, 73

N_COLD_AISLE = 12
N_HOT_AISLE = 2
N_ACU = 9
N_NODES = N_COLD_AISLE + N_HOT_AISLE + N_ACU
ACU_NODE_FEATS = 5
AISLE_NODE_FEATS = 2

OBS_DIM = 46
ACT_DIM = 18
U_DIM = 27
EXT_DIM = 8


class HetGNNEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.proj_aisle = nn.Linear(AISLE_NODE_FEATS, d_model)
        self.proj_acu   = nn.Linear(ACU_NODE_FEATS, d_model)

        self.type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(N_NODES, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        T = state[:, STATE_TEMPS_LO:STATE_TEMPS_HI].unsqueeze(-1)
        H = state[:, STATE_HUMID_LO:STATE_HUMID_HI].unsqueeze(-1)
        aisle_feats = torch.cat([T, H], dim=-1)
        aisle_emb = self.proj_aisle(aisle_feats) + self.type_embed[0]

        LAT = state[:, STATE_LAT_LO:STATE_LAT_HI].unsqueeze(-1)
        EAT = state[:, STATE_EAT_LO:STATE_EAT_HI].unsqueeze(-1)
        ONOFF = state[:, STATE_ONOFF_LO:STATE_ONOFF_HI].unsqueeze(-1)
        FAN = state[:, STATE_FAN_LO:STATE_FAN_HI].unsqueeze(-1)
        VALVE = state[:, STATE_VALVE_LO:STATE_VALVE_HI].unsqueeze(-1)
        acu_feats = torch.cat([LAT, EAT, ONOFF, FAN, VALVE], dim=-1)
        acu_emb = self.proj_acu(acu_feats) + self.type_embed[1]

        all_nodes = torch.cat([aisle_emb, acu_emb], dim=1)
        all_nodes = all_nodes + self.pos_embed.unsqueeze(0)
        return self.encoder(all_nodes)


class ExogenousEncoder(nn.Module):
    def __init__(self, in_dim: int = EXT_DIM, d_ext: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_ext), nn.LayerNorm(d_ext), nn.GELU(),
            nn.Linear(d_ext, d_ext), nn.LayerNorm(d_ext), nn.GELU(),
        )

    def forward(self, ext: torch.Tensor) -> torch.Tensor:
        return self.net(ext)


class WorldModelMember(nn.Module):

    def __init__(self, d_model: int = 128, d_ext: int = 64, gnn_layers: int = 2,
                 n_heads: int = 4, gru_hidden: int = 256, head_hidden: int = 256,
                 dropout: float = 0.0,
                 min_logvar_init: float = -10.0, gap_init: float = 10.0):
        super().__init__()
        self.encoder = HetGNNEncoder(d_model, gnn_layers, n_heads, dropout)
        self.ext_encoder = ExogenousEncoder(EXT_DIM, d_ext)

        self.gru = nn.GRUCell(d_model + d_ext + ACT_DIM, gru_hidden)
        self.gru_hidden_size = gru_hidden

        dec_in = gru_hidden + U_DIM + ACT_DIM + d_ext
        self.aisle_head = nn.Sequential(
            nn.Linear(dec_in, head_hidden), nn.LayerNorm(head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, (N_COLD_AISLE + N_HOT_AISLE) * 2),
        )
        self.acu_head = nn.Sequential(
            nn.Linear(dec_in, head_hidden), nn.LayerNorm(head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, N_ACU * 2),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(dec_in, head_hidden), nn.LayerNorm(head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, OBS_DIM),
        )

        self.min_logvar_base = nn.Parameter(torch.full((OBS_DIM,), float(min_logvar_init)))
        gap_base_init = torch.log(torch.expm1(torch.tensor(float(gap_init))))
        self.gap_base = nn.Parameter(torch.full((OBS_DIM,), float(gap_base_init.item())))

    @property
    def min_logvar(self):
        return self.min_logvar_base

    @property
    def max_logvar(self):
        return self.min_logvar_base + torch.nn.functional.softplus(self.gap_base)

    def _clamp_logvar(self, logvar):
        max_lv = self.max_logvar; min_lv = self.min_logvar
        logvar = max_lv - torch.nn.functional.softplus(max_lv - logvar)
        logvar = min_lv + torch.nn.functional.softplus(logvar - min_lv)
        return logvar

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.gru_hidden_size, device=device)

    def step(self, state, action_prev, action_cur, ext, h_prev):
        e_obs = self.encoder(state).mean(dim=1)
        e_ext = self.ext_encoder(ext)
        fused = torch.cat([e_obs, e_ext, action_prev], dim=-1)
        h_next = self.gru(fused, h_prev)

        u_t = state[:, STATE_U_LO:STATE_U_HI]
        dec_in = torch.cat([h_next, u_t, action_cur, e_ext], dim=-1)
        aisle_out = self.aisle_head(dec_in)
        acu_out = self.acu_head(dec_in)

        mean = torch.cat([
            aisle_out[:, :N_COLD_AISLE + N_HOT_AISLE],
            aisle_out[:, N_COLD_AISLE + N_HOT_AISLE:],
            acu_out[:, :N_ACU],
            acu_out[:, N_ACU:],
        ], dim=-1)

        logvar = self._clamp_logvar(self.logvar_head(dec_in))
        return mean, logvar, h_next


class BOREASEnsemble(nn.Module):
    def __init__(self, n_ensemble: int = 5, d_model: int = 128, d_ext: int = 64,
                 gnn_layers: int = 2, n_heads: int = 4,
                 gru_hidden: int = 256, head_hidden: int = 256):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.members = nn.ModuleList([
            WorldModelMember(d_model, d_ext, gnn_layers, n_heads,
                             gru_hidden, head_hidden)
            for _ in range(n_ensemble)
        ])
        self.gru_hidden_size = gru_hidden

    def init_hidden(self, batch_size, device):
        return [m.init_hidden(batch_size, device) for m in self.members]

    def step(self, state, action_prev, action_cur, ext, hidden_list):
        means, logvars, new_h = [], [], []
        for m, h in zip(self.members, hidden_list):
            mu, lv, hn = m.step(state, action_prev, action_cur, ext, h)
            means.append(mu); logvars.append(lv); new_h.append(hn)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0), new_h

    def gaussian_nll(self, means, logvars, target):
        tgt = target.unsqueeze(0).expand_as(means)
        inv_var = torch.exp(-logvars)
        return (0.5 * ((means - tgt) ** 2 * inv_var + logvars)).mean()

    def logvar_bound_penalty(self, weight: float = 0.01):
        acc = torch.zeros((), device=self.members[0].min_logvar_base.device)
        for m in self.members:
            acc = acc + (m.max_logvar - m.min_logvar).sum()
        return weight * acc


def state_next_from_delta_obs(state, action_raw, delta_obs):
    new_state = torch.empty_like(state)
    new_state[:, :OBS_DIM] = state[:, :OBS_DIM] + delta_obs
    new_state[:, STATE_ONOFF_LO:STATE_ONOFF_HI] = state[:, STATE_ONOFF_LO:STATE_ONOFF_HI]
    new_state[:, STATE_FAN_LO:STATE_FAN_HI] = state[:, STATE_FAN_LO:STATE_FAN_HI] + action_raw[:, :N_ACU]
    new_state[:, STATE_VALVE_LO:STATE_VALVE_HI] = state[:, STATE_VALVE_LO:STATE_VALVE_HI] + action_raw[:, N_ACU:]
    return new_state
