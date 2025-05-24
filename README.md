# ExplorAct

This repository provides implementation details and experiments for ExplorAct paper submission to CIKM 2025.

## 📦 Setup Instructions

### 1. Python Environment

Ensure Python **3.11.9** is installed. Then, install the required dependencies using:

```
pip install -r requirements.txt
```

### 2. Repository Structure

This repository includes a local clone of [REACT](https://github.com/TAU-DB/REACT-IDA-Recommendation-benchmark), so **no manual cloning is required**.

Before running analysis/code, **extract the `content.zip` files** in the following folders:

- `chunk_ted_results/`
- `chunked_sessions/`
- `dst_probs/`
- `model_stats/`

Each contains data necessary for evaluating different model variants.

---

## 📊 Accuracy and Plot Generation

To compute final accuracy scores and generate evaluation plots:

- Use `result_analyser.ipynb` for:
  - GRU
  - GINE
  - REACT
  - EA-SP
  - EA-MP

- Use `evidence_fusion.ipynb` for:
  - EF-SP
  - EF-MP

These Jupyter notebooks are **self-contained** with all required instructions in comments.

Already generated plots are available under the `time_plots/` directory.

---

## 🚀 Running Experiments

### REACT

To run the REACT implementation and calculate accuracy:

```
python react.py {seed} {size} {test_id}
```

To compute execution time against session log size:

```
python react_time_logsize.py {seed} {size}
```

---

### ExplorAct Models (EA-SP, EA-MP)

To run **accuracy experiments**:

```
python {file_prefix}.py {task} {seed} {size} {test_id}
```

Where `{file_prefix}` is one of:
- `ea_sp`
- `ea_mp`

By default, these scripts use GPU. To switch to CPU, edit the line in the corresponding file:

```python
torch.device('<device>')  # e.g., 'cpu'
```

---

### ExplorAct Inference Timing

#### CPU Inference Times

```
python {file_prefix}.py {task} {size}
```

Where `{file_prefix}` is:
- `ea_sp_time`
- `ea_mp_time`

#### Inference Time vs Session Log Size

```
python {file_prefix}.py {seed} {size}
```

Where `{file_prefix}` is:
- `ea_sp_time_logsize`
- `ea_mp_time_logsize`

---

## 🧪 Parameters

- `task` options:
  - `act` for τ-rec
  - `col` for a-rec
  - `tg` for (τ, a)-rec

- `seed` options: `20250212`, `20250214`, `20250314`
- `size` options: `3`, `4`, ..., `8`
- `test_id` options: `0`, `1`, ..., `4`

---

## 📁 Output
 
- Raw Results: `model_stats/` and `dst_probs` 
- Results analysis: from Jupyter notebooks  
- Plots: `time_plots/`

---

## 🔧 Notes

- As part of CIKM 2025 paper submission.
