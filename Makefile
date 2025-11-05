export PYTHONPATH := $(PWD)

# ==== Config ====
CONFIG    ?= configs/config.yaml
PY        ?= python
PIP       ?= pip
GPUS      ?= 2

REQ_BASE  ?= requirements.base.txt
REQ_CPU   ?= requirements.torch.cpu.txt
REQ_ROCM  ?= requirements.torch.rocm.txt

SHELL := /bin/bash
.ONESHELL:

# ==== Help ====
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make install            # (varsayılan) CPU/CUDA PyPI profili (base + cpu)"
	@echo "  make install-cpu        # CPU/CUDA PyPI profili (base + cpu)"
	@echo "  make install-amd        # AMD ROCm 6.4 profili (base + rocm)"
	@echo "  make env                # Python/pip/torch ve cihaz bilgisi"
	@echo "  make cuda-info          # CUDA/ROCm bilgisi (scripts/cuda_info.py)"
	@echo "  make train              # modeli eğit (--config=$(CONFIG))"
	@echo "  make eval               # test setinde değerlendirme (--config=$(CONFIG))\n\t@echo "  make backtest           # walk-forward backtest (PnL/Sharpe)""
	@echo "  make tb                 # TensorBoard aç (artifacts/tb)"
	@echo "  make ddp GPUS=N         # torchrun ile DDP eğitim"
	@echo "  make streamlit          # Streamlit arayüzünü başlat"
	@echo "  make freeze             # requirements-lock.txt üret"
	@echo "  make clean              # çıktı/önbellek temizle"
# ==== Setup ====
.PHONY: install
install: install-cpu

.PHONY: install-cpu
install-cpu:
	$(PIP) install -r $(REQ_BASE)
	$(PIP) install -r $(REQ_CPU)

.PHONY: install-amd
install-amd:
	$(PIP) install -r $(REQ_BASE)
	$(PIP) install -r $(REQ_ROCM)

# ==== Env / CUDA-ROCm Info ====
.PHONY: env
env:
	$(PY) -V
	$(PIP) -V
	$(PY) -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available(), 'hip:', getattr(torch.version,'hip', None))"

.PHONY: cuda-info
cuda-info:
	$(PY) scripts/cuda_info.py

# ==== Train / Eval ====
.PHONY: train
train:
	$(PY) -m src.cli --config $(CONFIG) --mode train

.PHONY: eval
eval:
	$(PY) -m src.cli --config $(CONFIG) --mode eval

# ==== TensorBoard ====
.PHONY: tb
tb:
	tensorboard --logdir artifacts/tb

# ==== DDP ====
.PHONY: ddp
ddp:
	torchrun --nproc_per_node=$(GPUS) -m src.cli --config $(CONFIG) --mode train --ddp 1

# ==== Streamlit ====
.PHONY: streamlit
streamlit:
	PYTHONPATH=. streamlit run streamlit_app/app.py

# ==== Utilities ====
.PHONY: freeze
freeze:
	$(PIP) freeze > requirements-lock.txt
	@echo "✔ requirements-lock.txt oluşturuldu."

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache **/__pycache__ artifacts/* *.egg-info
	@echo "✔ Temizlendi."


.PHONY: backtest
backtest:
	$(PY) -m src.cli --config $(CONFIG) --mode backtest


.PHONY: train-rocm-safe
train-rocm-safe:
	$(PY) -m src.cli --config configs/config.rocm-safe.yaml --mode train


.PHONY: report
report:
	$(PY) scripts/make_backtest_report.py


.PHONY: backtest-rocm-safe
backtest-rocm-safe:
	$(PY) -m src.cli --config configs/config.backtest.rocm-safe.yaml --mode backtest



.PHONY: streamlit-rocm-safe
streamlit-rocm-safe:
	# ROCm tarafında JIT'i sakinleştiren güvenli bayraklar
	# (gfx1030 için HSA_OVERRIDE_GFX_VERSION bazen gerekli)
	MIOPEN_FIND_MODE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
	PYTHONPATH=. streamlit run streamlit_app/app.py -- \
	  --config configs/config.backtest.rocm-safe.yaml \
	  --rocm-safe
