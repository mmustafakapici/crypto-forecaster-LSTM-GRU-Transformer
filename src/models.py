import os
import math
import torch
import torch.nn as nn

def _maybe_layernorm(norm, dim):
    return nn.LayerNorm(dim) if norm == "layernorm" else nn.Identity()

def _dprint(enabled: bool, *args):
    if enabled:
        print(*args)

class TickerEmbeddingMixin:
    """
    Ticker-id embedding sadece şu durumda aktiftir:
    add_ticker_id == True  VE  ticker_embedding_dim > 0
    Aktif değilse input'a dokunmuyoruz (feature sayısı değişmez).
    """
    def __init__(self, input_size, ticker_embedding_dim=0, add_ticker_id=False, dbg=False):
        # Embedding GERÇEKTEN aktif mi?
        self.use_ticker_emb = bool(add_ticker_id and ticker_embedding_dim > 0)
        self.ticker_embedding_dim = int(ticker_embedding_dim)

        # base_feat_dim: embedding AKTİFSE 1 eksilt (ticker id sütununu E-dim ile değiştireceğiz)
        # embedding PASİFSE eksiltme yok
        self.base_feat_dim = int(input_size - (1 if self.use_ticker_emb else 0))

        self._dbg_enabled = bool(dbg or os.getenv("DEBUG_SHAPES", "0") == "1")

        if self.use_ticker_emb:
            self.tok = nn.Embedding(256, self.ticker_embedding_dim)  # max 256 ticker

        _dprint(
            self._dbg_enabled,
            f"[TickerEmbeddingMixin] input_size={input_size} use_ticker_emb={self.use_ticker_emb} "
            f"ticker_emb_dim={self.ticker_embedding_dim} base_feat_dim={self.base_feat_dim}"
        )

    def fuse_feats(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F_in)
        if not self.use_ticker_emb:
            _dprint(self._dbg_enabled, f"[fuse_feats] (no-emb) x.shape={tuple(x.shape)}")
            return x

        base = x[..., :self.base_feat_dim]                      # (B,T,F_base)
        tidx = x[..., self.base_feat_dim].long().clamp(0, 255)  # (B,T)  (tek indeks!)
        emb  = self.tok(tidx)                                   # (B,T,E)
        fused = torch.cat([base, emb], dim=-1)                  # (B,T,F_base+E)

        _dprint(
            self._dbg_enabled,
            f"[fuse_feats] (with-emb) base={tuple(base.shape)} emb={tuple(emb.shape)} fused={tuple(fused.shape)}"
        )
        return fused

class LSTMForecast(TickerEmbeddingMixin, nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 norm=None, ticker_embedding_dim=0, add_ticker_id=False, dbg=False):
        nn.Module.__init__(self)
        TickerEmbeddingMixin.__init__(self, input_size, ticker_embedding_dim, add_ticker_id, dbg=dbg)

        self.eff_input = self.base_feat_dim + (self.ticker_embedding_dim if self.use_ticker_emb else 0)
        self.norm_in = _maybe_layernorm(norm, self.eff_input)
        self.lstm = nn.LSTM(
            input_size=self.eff_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

        _dprint(
            self._dbg_enabled,
            f"[LSTMForecast.__init__] eff_input={self.eff_input} hidden={hidden_size} layers={num_layers} dropout={dropout}"
        )

    def forward(self, x):
        _dprint(self._dbg_enabled, f"[LSTM.forward] in={tuple(x.shape)} expecting={self.eff_input}")
        x = self.fuse_feats(x)
        x = self.norm_in(x)
        _dprint(self._dbg_enabled, f"[LSTM.forward] after-fuse/norm={tuple(x.shape)}")
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUForecast(TickerEmbeddingMixin, nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 norm=None, ticker_embedding_dim=0, add_ticker_id=False, dbg=False):
        nn.Module.__init__(self)
        TickerEmbeddingMixin.__init__(self, input_size, ticker_embedding_dim, add_ticker_id, dbg=dbg)

        self.eff_input = self.base_feat_dim + (self.ticker_embedding_dim if self.use_ticker_emb else 0)
        self.norm_in = _maybe_layernorm(norm, self.eff_input)
        self.gru = nn.GRU(
            input_size=self.eff_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

        _dprint(
            self._dbg_enabled,
            f"[GRUForecast.__init__] eff_input={self.eff_input} hidden={hidden_size} layers={num_layers} dropout={dropout}"
        )

    def forward(self, x):
        _dprint(self._dbg_enabled, f"[GRU.forward] in={tuple(x.shape)} expecting={self.eff_input}")
        x = self.fuse_feats(x)
        x = self.norm_in(x)
        _dprint(self._dbg_enabled, f"[GRU.forward] after-fuse/norm={tuple(x.shape)}")
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerForecast(TickerEmbeddingMixin, nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, norm=None,
                 ticker_embedding_dim=0, add_ticker_id=False, nhead=4, dim_feedforward=256, activation="gelu", dbg=False):
        nn.Module.__init__(self)
        TickerEmbeddingMixin.__init__(self, input_size, ticker_embedding_dim, add_ticker_id, dbg=dbg)

        self.eff_input = self.base_feat_dim + (self.ticker_embedding_dim if self.use_ticker_emb else 0)
        self.proj = nn.Linear(self.eff_input, hidden_size)
        self.pos = PositionalEncoding(hidden_size)
        enc_layer = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward,
                                               dropout=dropout, batch_first=True, activation=activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm_out = _maybe_layernorm(norm, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        _dprint(
            self._dbg_enabled,
            f"[TransformerForecast.__init__] eff_input={self.eff_input} hidden={hidden_size} layers={num_layers} nhead={nhead}"
        )

    def forward(self, x):
        _dprint(self._dbg_enabled, f"[Transformer.forward] in={tuple(x.shape)} expecting={self.eff_input}")
        x = self.fuse_feats(x)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm_out(x)
        _dprint(self._dbg_enabled, f"[Transformer.forward] after-enc={tuple(x.shape)}")
        return self.fc(x[:, -1, :])

def build_model_from_cfg(input_size, cfg):
    m = cfg["model"]
    mtype = (m.get("type", "lstm") or "lstm").lower()
    dbg = bool(cfg.get("logging", {}).get("debug", False)) or (os.getenv("DEBUG_SHAPES","0") == "1")
    common = dict(
        input_size=input_size,
        hidden_size=m.get("hidden_size", 128),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("dropout", 0.2),
        norm=m.get("norm", None),
        ticker_embedding_dim=m.get("ticker_embedding_dim", 0),
        add_ticker_id=bool(cfg["data"].get("add_ticker_id", False)),
        dbg=dbg,
    )
    if mtype == "gru":
        return GRUForecast(**common)
    if mtype == "transformer":
        t = m.get("transformer", {}) or {}
        return TransformerForecast(**common, nhead=t.get("nhead",4),
                                   dim_feedforward=t.get("dim_feedforward",256),
                                   activation=t.get("activation","gelu"))
    return LSTMForecast(**common)
