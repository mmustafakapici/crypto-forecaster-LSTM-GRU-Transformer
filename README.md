# Crypto Forecaster — Pro Level MVP

Çoklu parite (**multi-ticker**) destekli, `yfinance` + `PyTorch` tabanlı **zaman serisi tahmin** projesi.
LSTM/GRU/Transformer modelleri, **walk-forward backtest**, **PnL/Sharpe**, **Streamlit UI**, **DDP** ve **ROCm-safe** preset ile gelir.

---

## Öne Çıkanlar

* ✅ **Multi-ticker panel** (BTC-USD, ETH-USD, …) — tek panelde birleştirip eğitir/tahminler.
* ✅ **Model seçenekleri:** `lstm` (varsayılan), `gru`, `transformer`
* ✅ **Ticker-embedding** (isteğe bağlı) + **Al/Bekle/Sat** sinyal üretimi
* ✅ **Walk-forward backtest** + **PnL / Sharpe**
* ✅ **DDP (torchrun)** ile çoklu GPU eğitim
* ✅ **TensorBoard** logging
* ✅ `torch.compile` ile **opsiyonel hızlandırma** (uyumlu sürümlerde)
* ✅ **Streamlit** arayüzü (1/3/5 günlük horizon “yaklaşık” projeksiyonlar)
* ✅ **En iyi/son model** checkpoint: `artifacts/model.best.pt`, `model.last.pt`
* ✅ **ROCm-safe** preset (MIOpen/hiprtc derleme sorunlarını azaltır)

---

## Proje Ağacı (özet)

```
crypto-lstm-mvp/
├─ configs/
│  ├─ config.yaml
│  ├─ config.rocm-safe.yaml
│  └─ config.backtest.rocm-safe.yaml
├─ src/
│  ├─ cli.py                # Komut satırı arabirimi
│  ├─ data.py               # yfinance, panel oluşturma
│  ├─ datasets.py           # sequence windowing
│  ├─ models.py             # LSTM/GRU/Transformer + ticker-embedding
│  ├─ train_pipeline.py     # eğitim akışı
│  ├─ train_eval.py         # train/eval epoch
│  ├─ backtest.py           # walk-forward + PnL/Sharpe
│  ├─ backtest_pipeline.py  # backtest entry
│  ├─ signals.py            # sinyal ve PnL yardımcıları
│  └─ utils.py              # seed, device, logging yardımcıları
├─ streamlit_app/
│  ├─ app.py                # Streamlit girişi
│  ├─ config_loader.py      # config + ROCm-safe override
│  ├─ features.py           # RSI/SMA/Volatilite vb.
│  ├─ panel.py              # çoklu ticker paneli
│  ├─ predictor.py          # ölçekleme + rollout tahmin
│  ├─ plots.py              # grafikler (tarih eksenli)
│  └─ ui.py                 # arayüz bileşenleri
├─ artifacts/               # ckpt/metrics/tb (çalışırken oluşur)
├─ requirements.base.txt
├─ requirements.torch.rocm.txt
├─ Makefile
└─ README.md
```

---

## Kurulum

```bash
# CPU/CUDA (PyPI)
make install

# AMD ROCm 6.4 (PyTorch ROCm wheels)
make install-amd
```

> Not: `requirements.base.txt` CPU/CUDA/PyPI uyumlu paket aralığına göre sabitlendi. ROCm için `requirements.torch.rocm.txt` ayrıca kurulur (PyTorch, TorchVision, TorchAudio ROCm 6.4).

---

## Hızlı Başlangıç

```bash
# Tek GPU/CPU eğitim
make train

# Değerlendirme (valid/test)
make eval

# Çoklu GPU (DDP)
make ddp GPUS=4
```

**TensorBoard**:

```bash
make tb   # artifacts/tb altında
```

**Backtest (Walk-Forward) + PnL/Sharpe**:

```bash
make backtest
# ROCm-safe için:
make backtest-rocm-safe
```

**Streamlit UI**:

```bash
# Normal
make streamlit

# ROCm-safe (gfx1030 gibi kartlarda JIT’i sakinleştirir)
make streamlit-rocm-safe
```

---

## Streamlit’te Neler Görürsün?

* Parite listesi (virgülle): `BTC-USD,ETH-USD,SOL-USD`
* Tarih aralığı + feature zenginleştirme (SMA7, SMA21, VOL20, RSI14)
* Sequence uzunluğu (L)
* **1/3/5 günlük** horizon için **yaklaşık** getiri/fiyat projeksiyonu
* **Al/Bekle/Sat** sinyallerini renkli noktalarla “Pred vs True” grafiği üzerinde görürsün.
* **Equity Curve** ve test rollout için **Sharpe** hesapları.

> Not: Horizon projeksiyonları son barın tahminini **basit bileşik** varsayımıyla çoğaltır:
> ( \text{ret}*h \approx e^{h \cdot \hat{r}} - 1 ),  ( \hat{P}*{t+h} \approx P_t \cdot e^{h\cdot \hat{r}} ).
> Bu, tam çok-adımlı (multi-step) iteratif forecast değildir; UI’de hızlı fikir vermek amaçlıdır.

---

## Konfigürasyon

`configs/config.yaml` (örnek alanlar):

```yaml
data:
  tickers: ["BTC-USD", "ETH-USD", "SOL-USD"]
  start: "2018-01-01"
  end:   "2025-01-01"
  interval: "1d"
  features: ["Open", "High", "Low", "Close", "Volume"]
  add_ticker_id: true      # multi-ticker için ID ekler (embedding ile eşleşir)
  seq_len: 60

model:
  type: "lstm"             # lstm | gru | transformer
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  ticker_embedding_dim: 8  # >0 ve add_ticker_id:true ise aktif

train:
  epochs: 50
  batch_size: 64
  lr: 1e-3
  weight_decay: 1e-4
  mixed_precision: true
  compile: true            # torch.compile (uyumlu sürümlerde)
  gradient_clip: 1.0
  early_stop_patience: 8

strategy:
  threshold_long:  0.0005  # 5 bps
  threshold_short: 0.0005
  transaction_cost_bps: 5
  hold_flat: false

logging:
  tensorboard_dir: "artifacts/tb"
  debug: false

artifacts:
  dir: "artifacts"
  ckpt_best: "artifacts/model.best.pt"
  ckpt_last: "artifacts/model.last.pt"
  scaler: "artifacts/scaler.pkl"
  metrics: "artifacts/metrics.json"

dist:
  ddp: false
  seed: 42
```

### ROCm-Safe Preset

Bazı ROCm/MIOpen kurulumlarında LSTM + dropout / AMP derlemesi **hiprtc/MIOpen** hataları verebilir.
Güvenli profil:

```bash
make train-rocm-safe            # configs/config.rocm-safe.yaml
make backtest-rocm-safe
make streamlit-rocm-safe
```

Bu profil:

* `model.num_layers=1`, `model.dropout=0.0`
* `train.mixed_precision=false`, `train.compile=false`
* Gerekirse `MIOPEN_FIND_MODE=1` ve `HSA_OVERRIDE_GFX_VERSION=10.3.0` export edilir.

---

## Nasıl Çalışır? (Kısa teknik özet)

1. **Veri & Hedef:**
   OHLCV çekilir. Hedef: **ertesi bar log-getirisi**
   ( r_t = \log(\text{Close}*t / \text{Close}*{t-1}) )

2. **Özellikler:**
   Temel OHLCV + seçilebilir teknikler (SMA7/21, Vol20, RSI14). Multi-ticker panel **alt alta** birleştirilir. İstenirse **ticker ID** → embedding.

3. **Ölçekleme:**
   Yalnızca **train**’e `StandardScaler` fit → val/test transform.

4. **Pencereleme:**
   Uzunluk (L) sekans: ( X_{(t-L+1):t} \rightarrow y_{t+1} )

5. **Model:**
   LSTM/GRU/Transformer (embedding ile fused input). AMP/compile isteğe bağlı.

6. **Backtest:**
   Walk-forward bölmeler → testte ( \hat{r}_t ).
   Eşiklerle **long/flat/short** → **PnL**/**Sharpe**.

---

## Makefile Hedefleri

```bash
# Kurulum
make install
make install-amd

# Eğitim / Değerlendirme
make train
make eval

# Çoklu GPU (DDP)
make ddp GPUS=4

# Backtest
make backtest
make backtest-rocm-safe

# TensorBoard
make tb

# Streamlit
make streamlit
make streamlit-rocm-safe

# Dondur / Temizle
make freeze
make clean

# Ortam & ROCm bilgisi
make env
make cuda-info
```

---

## Troubleshooting

* **`attempted relative import with no known parent package`**
  Streamlit içindeki modüller **absolute import** kullanır (`from features import ...`).
  PYTHONPATH için `PYTHONPATH=.` kullanıyoruz (Makefile’da var).

* **ROCm/MIOpen `HIPRTC_ERROR_COMPILATION` / `miopenStatusUnknownError`**
  `make *-rocm-safe` hedeflerini kullanın. Gerekirse ortam bayrakları:

  ```bash
  export MIOPEN_FIND_MODE=1
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  ```

  Son çare olarak CPU’da denenebilir:

  ```bash
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" make streamlit
  ```

* **`input.size(-1) must be equal to input_size`**
  Bu, modelin beklediği feature sayısıyla gerçek girdi arasındaki uyumsuzluğu gösterir.
  Multi-ticker + embedding kullanırken **train** ve **inference**’ta **aynı feature seti** ve **add_ticker_id** durumuna sadık kalın.

---

## Çıktılar

* `artifacts/model.best.pt` — en iyi checkpoint
* `artifacts/model.last.pt` — son epoch checkpoint
* `artifacts/scaler.pkl` — train scaler
* `artifacts/metrics.json` — metrikler (MAE, dir_acc vb.)
* Backtest çıktıları: `artifacts/backtest/…` (Sharpe, equity, returns)

---

## Uyarı

Bu proje **MVP/AR-GE** amaçlıdır; çıktı sinyalleri **yatırım tavsiyesi değildir**.
Kripto piyasaları **yüksek volatilite** ve **rejim değişimleri** içerir.
Eşik, horizon, feature seti ve model seçimi sonuçları ciddi şekilde etkiler.

---

## Lisans

MIT (isteğe bağlı kendi lisans metninizle güncelleyebilirsiniz).
