export PYTHONPATH := $(PWD)

# ==== Venv / Python Ayarları ====
VENV     ?= .venv
PY       ?= $(VENV)/bin/python
PIP      ?= $(VENV)/bin/pip

# ==== Config ====
CONFIG    ?= configs/config.yaml
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
	@echo "  make venv               # .venv ortamını oluştur"
	@echo "  make install            # (varsayılan) venv + base + cpu kur"
	@echo "  make install-cpu        # venv + base + cpu profili"
	@echo "  make install-amd        # venv + base + rocm profili"
	@echo "  make env                # Python/pip/torch ve cihaz bilgisi"
	@echo "  make cuda-info          # CUDA/ROCm bilgisi (scripts/cuda_info.py)"
	@echo "  make train              # modeli eğit (--config=$(CONFIG))"
	@echo "  make eval               # test setinde değerlendirme (--config=$(CONFIG))"
	@echo "  make backtest           # walk-forward backtest (PnL/Sharpe)"
	@echo "  make tb                 # TensorBoard aç (artifacts/tb)"
	@echo "  make ddp GPUS=N         # torchrun ile DDP eğitim"
	@echo "  make streamlit          # Streamlit arayüzünü başlat"
	@echo "  make freeze             # requirements-lock.txt üret"
	@echo "  make clean              # çıktı/önbellek temizle"

# ==== Venv Oluşturma ====
.PHONY: venv
venv:
	@if command -v python3 >/dev/null 2>&1; then \
		PYBIN=python3; \
	elif command -v python >/dev/null 2>&1; then \
		PYBIN=python; \
	else \
		echo "❌ Python bulunamadı. Lütfen python3 kur."; \
		exit 1; \
	fi; \
	$$PYBIN -m venv $(VENV); \
	$(VENV)/bin/python -m pip install --upgrade pip; \
	echo "✔ Venv oluşturuldu: $(VENV)"
	

# ==== Setup ====
.PHONY: install
install: venv install-cpu

.PHONY: install-cpu
install-cpu:
	$(PIP) install -r $(REQ_BASE)
	$(PIP) install -r $(REQ_CPU)

.PHONY: install-amd
install-amd: venv
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

# ==== Train / Eval / Backtest ====
.PHONY: train
train:
	$(PY) -m src.cli --config $(CONFIG) --mode train

.PHONY: eval
eval:
	$(PY) -m src.cli --config $(CONFIG) --mode eval

.PHONY: backtest
backtest:
	$(PY) -m src.cli --config $(CONFIG) --mode backtest

# ==== TensorBoard ====
.PHONY: tb
tb:
	$(VENV)/bin/tensorboard --logdir artifacts/tb

# ==== DDP ====
.PHONY: ddp
ddp:
	$(VENV)/bin/torchrun --nproc_per_node=$(GPUS) -m src.cli --config $(CONFIG) --mode train --ddp 1

# ==== Streamlit ====
.PHONY: streamlit
streamlit:
	PYTHONPATH=. $(PY) -m streamlit run streamlit_app/app.py

# ==== Utilities ====
.PHONY: freeze
freeze:
	$(PIP) freeze > requirements-lock.txt
	@echo "✔ requirements-lock.txt oluşturuldu."

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache **/__pycache__ artifacts/* *.egg-info
	@echo "✔ Temizlendi."

.PHONY: report
report:
	$(PY) scripts/make_backtest_report.py
