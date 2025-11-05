# 1. Project context
- This repository follows the THUML **Time-Series-Library** layout: folders such as `data_provider/`, `exp/`, `models/`, `layers/`, `utils/`, plus launcher scripts (e.g., `run_longExp.py`) and experiment drivers (e.g., `exp/exp_long_term_forecasting.py`).
- Goal: multivariate time-series forecasting (long- and short-term) with reproducible training/evaluation and clear ablations.
- Target environment: Python >=3.9, PyTorch =1.13.1, CUDA when available, CPU fallback supported.

---

# 2. Repository structure
- `data_provider/`: dataset loaders, normalization, train/val/test split logic.
- `models/`: model definitions, each in a separate file or folder.
- `layers/`: reusable network modules (embeddings, attention/mamba layers, convolutional/projection blocks, normalization).
- `exp/`: experiment drivers for training, validation, testing.
- `utils/`: general utilities (logging, seeding, metrics, checkpointing).
- Top-level scripts (`run_longExp.py`, `run_shortExp.py`, etc.) serve as CLI entry points.

> If a folder is missing, reuse existing ones—do not create arbitrary new top-level directories.

---

# 3. Data and shapes
- Default tensor shape: **(B, L, D)** for inputs (batch, lookback length, feature dimension).  
- Output tensor shape: **(B, H, D′)** (forecast horizon and predicted variables).
- `seq_len` = lookback length, `label_len` = warm-up segment, `pred_len` = forecast horizon.
- Scaling must be **fit only on the training split**, then applied to val/test; data leakage is strictly prohibited.

---

# 4. Coding standards
- Use **Python type hints** and `from __future__ import annotations`.
- Follow **PEP 8**; format with `black` + `isort`, lint with `ruff`.
- Use **NumPy-style docstrings**, documenting input/output shapes and dtypes.
- Use the `logging` module (`logging.getLogger(__name__)`) instead of `print`.
- Centralize randomness in `utils/seed.py` (seed all libraries and workers).

---

# 5. Model contracts
Each model under `models/` must:
1. Define `__init__(self, cfg)` with all hyperparameters from the config.
2. Implement `forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` returning `(B, O, D′)`.
3. Avoid file I/O or command-line parsing within the model.
4. Optionally define a static method `add_model_specific_args(parser)` for CLI registration.
5. Support parameter counting (and optionally FLOPs reporting).

---

# 6. Training and evaluation rules
- Default optimizer: **AdamW** with configurable weight decay.
- LR schedulers: StepLR or CosineAnnealingLR.
- Support **mixed precision** (`torch.cuda.amp.autocast`), configurable via `--precision` or `--amp`.
- Implement **early stopping** based on validation metric, with configurable `--patience`.
- Metrics: MSE and MAE; define them in `utils/metrics.py`.
- Ensure reproducibility: log `seed`, library versions, device info, and environment.

---

# 7. Command-line and configs
- Common arguments:
  - Data: `--data`, `--features`, `--seq_len`, `--label_len`, `--pred_len`
  - Model: `--model`, `--enc_in`, `--dec_in`, `--c_out`
  - Training: `--batch_size`, `--lr`, `--epochs`, `--patience`, `--lradj`
  - Device & logging: `--use_gpu`, `--device`, `--exp_id`, `--save_dir`, `--seed`
- Example run:
  ```bash
  python run_longExp.py     --model DynamicMLP --data ETTm1 --features M     --seq_len 96 --label_len 48 --pred_len 96     --enc_in 7 --dec_in 7 --c_out 7     --batch_size 32 --lr 1e-3 --epochs 10 --exp_id demo
  ```

---

# 8. What Copilot should generate
When prompted to generate new components, follow these rules:

### New model
- File path: `models/<ModelName>.py`
- Must include:
  - Config-driven `__init__`
  - Documented `forward` method accepting `(B, L, D)` and returning `(B, H, D′)`
  - Optional shape assertions for debugging
- Add a lightweight unit test under `tests/` verifying:
  - Forward pass output shape and reproducibility under fixed seed
  - Simple one-step training run (<10s on CPU)
- Integrate model registration in a single factory function (e.g., `models/__init__.py`).

### New layer
- Place reusable blocks under `layers/`, and import them from models.

### Data transform
- Place normalization, padding, slicing in `data_provider/` or `utils/`.

### Config
- Add YAML files in `configs/` with only model-specific overrides; do not duplicate defaults.

---

# 9. What to avoid
- ❌ Fitting normalization on val/test splits  
- ❌ Using pandas in the training loop (use PyTorch/NumPy)  
- ❌ Hard-coded file paths or constants  
- ❌ Global singletons (except logger/config)

---

# 10. Logging and outputs
- Default logging via Weights & Biases **(offline mode)** or simple CSV/JSON.
- Save artifacts under:
  ```
  results/{dataset}/{model}/{exp_id}/
  ```
- Include:
  - Best checkpoint (`best.pth`)
  - Last checkpoint (`last.pth`)
  - Config dump and metrics log
  - Model summary (#params, FLOPs if available)

---

# 11. Performance tips for MTSF
- Use MAE for ETT/Weather; for Traffic/ECL, handle outliers (robust or quantile losses).
- FFT-based seasonal features are optional; keep modular.
- Keep `num_workers` and `pin_memory` configurable in dataloaders.

---

# 12. Quick checklists for generated code
- [ ] Model docstring clearly documents shapes and example usage.  
- [ ] `forward()` returns `(B, H, D′)` and preserves dtype.  
- [ ] Unit test passes on CPU in <10s.  
- [ ] Runner can recognize the model automatically (no multiple file edits).  

---

# 13. Glossary
| Symbol     | Meaning |
|------------|----------|
| L          | Input/Lookback window length |
| H          | Forecast horizon length |
| D          | Number of variables/features (encoder) |
| D′ / c_out | Number of predicted variables |
| MS / M / S | Feature modes: multivariate-to-single, multivariate, single-variable |
| seq_len    | Lookback length |
| pred_len   | Prediction horizon |
| label_len  | Warm-up length (used in encoder-decoder models) |

---

# 14. Summary
This file defines the **rules, structure, and expectations** for code generation within this repository.  
GitHub Copilot will read it to:
- follow your file organization (`models/`, `layers/`, `utils/`, etc.),
- respect your data format and type conventions,
- generate scientifically correct and reproducible PyTorch code,
- and avoid bad practices like data leakage or inconsistent model interfaces.
