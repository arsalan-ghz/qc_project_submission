# qc_project

Hybrid classical + quantum residual learning project inspired by QCardEst/QCardCorr-style post-processing.

At a high level:
- Build an object-level dataset from AGNmass reverberation measurements.
- Train a classical Ridge baseline.
- Train a quantum (Qiskit ML) model to predict the residual $\Delta = y - \hat{y}_\mathrm{base}$ and form $\hat{y}_\mathrm{quant} = \hat{y}_\mathrm{base} + \hat{\Delta}$.

## Repository structure (what to look at)
- Data acquisition + preparation:
   - `bulk_extract_reverb_with_metadata.py`: downloads AGNmass HTML tables and writes per-object CSVs.
   - `qc_data_preprocessing.ipynb`: cleans/normalizes the raw tables into the finalized dataset(s).
   - `draft.ipynb`: patches a known parsing issue in `data/qc_finalized_data.csv` and writes `data/qc_finalized_data_patched.csv`.
   - `notebooks/01_build_object_dataset.ipynb`: builds the object-level modeling table `data/model_dataset_object_level.csv`.
- Training + evaluation:
   - `run_qcardest_style_training.py`: main end-to-end runner (Ridge baseline + quantum residual correction) that writes metrics/predictions CSVs under `results/`.
   - `run_qcardest_style_sweep.py`: optional sweep/benchmark wrapper around the training runner.
- Core quantum implementation:
   - `src/qml_model.py`: builds the Qiskit ML `SamplerQNN` and wraps it as a Torch module.
   - `src/qml_layers.py`: QCardEst-style reduction + normalization layers (and optional bitstring selection).
   - `src/qml_interpretation.py`: QCardEst-style “heads” (`linear`, `rational`, `rationalLog`) mapping reduced outputs to a scalar correction.
   - `src/qc_quantum_corr.py`: angle encoding utilities (`z`-scores → clipped/scaled rotation angles).
- Artifacts + references:
   - `results/`: saved metrics and prediction CSVs.

## Quantum logic summary (the part this project is about)

This repo implements a QCardEst-style **quantum residual correction** model.

**1) Classical baseline (Ridge)**
- In `run_qcardest_style_training.py`, a Ridge pipeline (median impute → standardize → Ridge) predicts $\hat{y}_\mathrm{base}$.

**2) Residual learning target**
- Define $\Delta = y - \hat{y}_\mathrm{base}$ and train the quantum model to predict $\hat{\Delta}$.
- Final prediction is composed as $\hat{y}_\mathrm{quant} = \hat{y}_\mathrm{base} + \hat{\Delta}$.

**3) Feature-to-angle encoding**
- Features are standardized (z-scores), clipped, and mapped into rotation angles via `src/qc_quantum_corr.py` (`AngleMap`, `angles_from_z`).

**4) QNN base model (Qiskit ML → Torch)**
- `src/qml_model.py` builds a parameterized VQC, measures all qubits, and uses Qiskit Machine Learning’s `SamplerQNN` to return a probability vector over basis states (shot-based sampling).
- `TorchConnector` wraps the QNN as a Torch layer so we can train with standard optimizers (Adam) and backprop.

**5) Output reduction + normalization (QCardEst-style)**
- The raw probability vector has dimension $2^n$. We reduce it before interpretation:
   - default `--readout sum`: `ReshapeSumLayer(n_outputs)` bins + sums the vector into `n_outputs` components.
   - optional `--readout bitstring`: `BitstringSelectLayer(bitstrings)` selects explicit basis states.
- Optional `NormLayer()` applies per-sample max-normalization $x / x_\max$ (QCardEst-style; not sum-to-1).

**6) Interpretation heads (map vector → scalar correction)**
- `src/qml_interpretation.py` applies one of:
   - `linear`: learned scale on the first reduced component (no bias)
   - `rational`: $(x_0+\epsilon)/(x_1+\epsilon)$
   - `rationalLog`: $\log\big((x_0+\epsilon)/(x_1+\epsilon)\big)$

**7) Training loop + reproducibility**
- `run_qcardest_style_training.py` trains the Torch model on MSE for $\Delta$ and records both baseline and corrected metrics.
- Seeds are passed separately for split/training/QNN sampling so results are attributable and reproducible across runs.

## Environment setup

### Recommended: Conda env (Python 3.10)
The pinned Qiskit stack (e.g. `qiskit==0.39.2`, `qiskit-aer==0.11.1`) only ships wheels for CPython 3.10. On newer interpreters (3.11/3.12) `pip` typically falls back to building from source, which pulls in a legacy Conan/CMake toolchain that often fails. To match the working environment:

```bash
conda create -y -n qc_project python=3.10
conda activate qc_project
pip install -r requirements.txt
# optional GPU extras
pip install -r requirements.cuda.txt
```

Notes:
- `requirements.txt` matches the working conda environment and keeps CUDA wheels separate.
- `pandas` (data loading) and `tqdm` (progress bars) are included so CLI scripts run in fresh envs.
- The CUDA extras are pure wheels that bundle cu12 runtimes; install only if you need GPU execution.

### If you must use `python -m venv`
Stick to Python 3.10 to stay on the published wheels:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements.cuda.txt  # optional
```

Trying to install with Python 3.11+ will likely force a source build of `qiskit-aer` and require a working C/C++ toolchain + legacy packaging combinations.

## Data pipeline

1. Extract AGNmass reverberation-measurement tables (with object-level metadata appended as columns):

   ```bash
   python3 bulk_extract_reverb_with_metadata.py --start 1 --end 95
   ```

   Notes:
   - The script defaults to `--sep ';'` (recommended). Avoid `--sep ','` because it can break downstream notebook parsing.
2. Run all cells in `qc_data_preprocessing.ipynb` to generate the cleaned/normalized dataset(s).
3. In `draft.ipynb`, run the last cell to produce `data/qc_finalized_data_patched.csv`.
4. Run `notebooks/01_build_object_dataset.ipynb` to produce `data/model_dataset_object_level.csv`.

## Training and outputs

### Single run (baseline + quantum correction)
`run_qcardest_style_training.py` runs the full pipeline and writes two CSVs:
- metrics: `results/qcardest_style_metrics.csv`
- predictions: `results/qcardest_style_predictions.csv`

Example:

```bash
python run_qcardest_style_training.py --epochs 25 --shots 200 --reps 2 --seed 42
```

### Readout mode notes
- Default readout is the legacy *sum* reduction (reshape to `n_outputs` buckets and sum).
- Switch to explicit bitstring selection with `--readout bitstring` to select basis states (e.g. `000…0` / `111…1`) before the interpretation layer.
- When using `--readout bitstring`, override selected states via `--bitstrings`, e.g. `--bitstrings 000 011 101` when `--n-outputs 3`. The list length must equal `n_outputs` and each entry must match the circuit width (number of qubits).

## Notes

## Quantum logic breakdown (implementation map)

This section provides a structured, step-by-step breakdown of the quantum logic and points to the exact scripts and functions that implement each stage. Use it as a reference to understand how the pipeline is realized in code.

### Overview (helicopter view)
- `run_qcardest_style_training.py`: Orchestrates the end-to-end flow (data split, baseline Ridge, residual target, QNN training, and baseline+correction composition).
- `src/qc_quantum_corr.py`: Feature-to-angle utilities (z-score clipping and scaling into rotation angles).
- `src/qml_model.py`: Builds the variational circuit, sampler-backed QNN, and readout stack (reshape-sum or bitstring selection + optional normalization).
- `src/qml_layers.py`: Reduction and normalization layers that implement QCardEst-style post-processing.
- `src/qml_interpretation.py`: Interpretation heads (`linear`, `rational`, `rationalLog`) that map reduced outputs to a scalar correction.

The sections below provide a step-by-step mapping from the conceptual pipeline to the exact functions and code paths.

### 1. Standardize features → build z
- **File:** `run_qcardest_style_training.py`, function `_run_one_experiment`
- **Code:**
   - `scaler_q = Pipeline([... SimpleImputer, StandardScaler ...])`
   - `Z_train = scaler_q.fit_transform(X_train)` / `Z_test = scaler_q.transform(X_test)`
- **Role:** computes per-feature z-scores $z_j = (x_j-\mu_j)/\sigma_j$ on the quantum branch.

### 2. Clip and scale → angles φ
- **Files:**
   - `src/qc_quantum_corr.py` defines `AngleMap(clip, scale)` and `angles_from_z(z, angle_map)`.
   - `run_qcardest_style_training.py` calls `angles_from_z` for every z-vector (`Xang_train = ...`).
- **Role:** clamps each z component to $[-\text{clip}, \text{clip}]$ and multiplies by `scale` ($s = \pi/2$ by default) to produce the rotation angles $\phi_j$.

### 3. Angle encoding gates RY(φ)
- **File:** `src/qml_model.py`, helper `_build_parametrized_vqc`.
- **Code:** `qc.ry(x_params[i], i)` for every qubit `i`.
- **Role:** applies the data-dependent RY(φ) rotations that inject the preprocessed features into the circuit.

### 4. Trainable variational block (RY/RZ per qubit, repeated reps times)
- **File:** `src/qml_model.py`, `_build_parametrized_vqc`.
- **Code:** inside the `for _ in range(reps)` loop the circuit applies `qc.ry(theta_params[idx], i)` and `qc.rz(...)` on every qubit.
- **Control:** `_run_one_experiment` builds `QNNSettings(reps=args.reps, ...)`, so the CLI `--reps` flag selects how many times this block repeats.

### 5. Entanglement (CX chain)
- **File:** `src/qml_model.py`, `_build_parametrized_vqc`.
- **Code:** `for i in range(n_qubits - 1): qc.cx(i, i + 1)` inside each repetition.
- **Role:** implements the CX chain (0→1, 1→2, …) to introduce feature interactions.

### 6. Measurement + sampler back-end
- **File:** `src/qml_model.py`.
   - `_build_parametrized_vqc` ends with `qc.measure_all()` so that the sampler sees measured bitstrings.
   - `build_qnn_base_model` wraps the circuit in a `SamplerQNN` (`TorchConnector`) with an Aer sampler configured via `QNNSettings(shots, seed, readout, ...)`.
- **Role:** this is the “measure all qubits → probability distribution → Torch layer” portion of the QNN pipeline.

### QCardEst-style reduction + interpretation

#### 7. Raw probability vector → reduced outputs
- **File:** `src/qml_model.py`, function `build_qnn_base_model`
- **Code path:**
   - `SamplerQNN(..., interpret=lambda i: i, output_shape=2**n_qubits)` produces the full probability vector.
   - readout selection:
      - `ReshapeSumLayer(n_outputs)` when `readout == "sum"`
      - `BitstringSelectLayer(bitstrings)` when `readout == "bitstring"`
- **Role:** implements QCardEst’s post-measurement reduction of the $2^n$-length vector into `n_outputs` components.

#### 8. Reshape+sum reduction (legacy/QCardEst)
- **File:** `src/qml_layers.py`, class `ReshapeSumLayer`
- **Role:** reshapes the last dimension into `(n_outputs, -1)` and sums each bin, matching the “reshape + sum” description.

#### 9. Explicit bitstring reduction (optional)
- **File:** `src/qml_layers.py`, class `BitstringSelectLayer`
- **Role:** selects specific computational-basis indices by bitstring (e.g., `000...0`, `111...1`) instead of bin-summing.
- **Configuration:** set `--readout bitstring` and optionally provide `--bitstrings` in `run_qcardest_style_training.py`.

#### 10. Max-normalization (QCardEst-style)
- **File:** `src/qml_layers.py`, class `NormLayer`
- **Role:** divides each output vector by its per-sample maximum (`x / max(x)`), matching the normalization step (not sum-to-1 normalization).
- **Configuration:** controlled by `QNNSettings(norm=...)`, set via `--norm/--no-norm` in `run_qcardest_style_training.py`.

#### 11. Interpretation heads (map reduced vector → scalar Δ̂)
- **File:** `src/qml_interpretation.py`
- **Classes/behavior:**
   - `LinearScale`: learned scalar on the first component (no bias)
   - `Rational`: $(x_0+\epsilon)/(x_1+\epsilon)$
   - `RationalLog`: $\log((x_0+\epsilon)/(x_1+\epsilon))$
- **Factory:** `from_string(name, base_model)` selects the head based on `--archs`.
- **Invocation:** `run_qcardest_style_training.py` wraps the base QNN with `from_string(arch, base_qnn)` for each architecture.

### Training, optimization, and correction composition

#### 12. QNN parameter learning (optimization loop)
- **File:** `run_qcardest_style_training.py`, function `_train_one`
- **Code path:**
   - Builds a Torch `DataLoader` for mini-batches.
   - Uses `torch.optim.Adam` and `torch.nn.MSELoss()`.
   - Calls `loss.backward()` and `opt.step()` to update circuit parameters exposed by `TorchConnector`.
- **Role:** performs gradient-based optimization over the circuit weights (and the linear head’s scalar when `arch=linear`).

#### 13. Train/evaluate orchestration per architecture
- **File:** `run_qcardest_style_training.py`, function `_run_one_experiment`
- **Code path:**
   - `base_qnn = build_qnn_base_model(...)` builds the QNN + reduction stack.
   - `model = from_string(arch, base_qnn)` attaches the chosen interpretation head.
   - `_train_one(...)` trains the model for the configured epochs.
- **Role:** loops over architectures (`--archs`) so each head is trained and evaluated separately.

#### 14. Baseline + correction composition
- **File:** `run_qcardest_style_training.py`, function `_run_one_experiment`
- **Code path:**
   - Baseline: `y_base_test = base_model.predict(X_test)`
   - Correction: `delta_hat_test = result["pred_test"]`
   - Final prediction: `y_pred_quant = y_base_test + delta_hat_test`
- **Role:** combines the learned residual with the classical baseline to form the final corrected prediction.