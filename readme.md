# 🫀 Pre-Apnea Detection from ECG Signals using CNN-Transformer

> Predicting sleep apnea **before it happens** using a single-lead ECG signal and deep learning.

---

##  Overview

Sleep apnea is a condition where a person **stops breathing repeatedly during sleep**. It affects over 425 million people worldwide and can lead to serious health problems like heart disease and stroke.

Most existing systems detect apnea **while it is already happening** — which is too late to intervene. This project takes a different approach:

> **Instead of detecting apnea during the event, we predict it ~45 seconds before it occurs.**

We train a deep learning model (CNN + Transformer) on overnight ECG recordings to recognize subtle heart signal patterns that appear just before a breathing stoppage — patterns invisible to the human eye but learnable by AI.

---

##  What It Does

### The Core Idea

```
Normal approach:    Patient stops breathing → alarm → too late ❌
Our approach:       ECG shows early signs  → alarm → doctor acts → breathing stops prevented ✅
```

The model watches a 60-second window of ECG signal every minute and answers one question:

> **"Is this person about to stop breathing in the next ~45 seconds?"**

---

### Pipeline at a Glance

```
PhysioNet ECG Dataset (35 patients, 8hr overnight recordings)
        │
        ▼
Sliding Window (60s chunks)
        │
        ▼
Label each chunk:
  → apnea   (breathing stopped)
  → prior   (looks normal, but apnea coming soon) ← key innovation
  → normal  (truly normal)
        │
        ▼
Train CNN-Transformer on "prior vs normal"
        │
        ▼
Model learns: "what does ECG look like BEFORE apnea?"
        │
        ▼
On new patient → fires alert 45 seconds early [🚨]
```

---

### Key Innovation — "Prior" Label

The dataset only comes with two labels: **apnea** and **normal**.

We introduce a third label — **prior** — by walking backwards from every apnea event and relabeling the normal windows just before it:

```
[normal] [normal] [PRIOR] [PRIOR] [APNEA] [APNEA]
                    ↑       ↑
              model learns these patterns
              even though they look "normal" to a human
```

This is what makes early prediction possible — the model is trained on what ECG looks like **before** apnea, not during it.

---

## Dataset

**PhysioNet Apnea-ECG Database** (Penzel et al., 2000)

| Property | Detail |
|---|---|
| Patients | 35 overnight recordings |
| Signal | Single-lead ECG |
| Sampling Rate | 100 Hz |
| Labels | Per-minute A (apnea) / N (normal) by sleep experts |
| Access | Free, no login needed |
| Source | [physionet.org/content/apnea-ecg](https://physionet.org/content/apnea-ecg/1.0.0/) |

Labels were assigned by clinical experts watching **breathing sensor data** overnight — not ECG. This means our model learns to predict apnea from heart signals alone, without needing any breathing sensor at inference time.

---

## Model Architecture

We use a **CNN-Transformer hybrid** model:

```
Input: (batch, 1, 4000)   ← 40s of ECG at 100Hz
         │
         ▼
CNN Block 1 (kernel=3)    ← learns sharp heartbeat shapes
         │
CNN Block 2 (kernel=29)   ← learns rhythm patterns
         │
CNN Block 3 (kernel=29)   ← expands to 64 feature channels
         │
MaxPool (÷10)             ← compresses sequence, saves memory
         │
Transformer (2L, 8H)      ← understands how patterns evolve over time
         │
Average Pool              ← summarizes entire window
         │
Dropout + Linear          ← final classification
         │
         ▼
Output: probability of apnea in next ~45 seconds
```

**Why CNN + Transformer?**
- CNN alone: sees local heartbeat patterns but misses long-range changes
- Transformer alone: too memory-heavy on raw ECG (6000×6000 attention matrix)
- Together: CNN extracts features first, Transformer models their evolution over time ✅

---

## Results

Evaluated using 5-fold stratified cross-validation:

| Metric | Score |
|---|---|
| Accuracy | ~87% |
| Sensitivity | ~92% |
| F1 Score | ~84% |

**Sensitivity of 92%** means the model catches 9 out of 10 pre-apnea events — the most clinically important number, since missing a warning is worse than a false alarm.

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/priyanshu026922/Sleep-Apnea-Detection.git
```

### 2. Install dependencies
```bash
pip install wfdb numpy torch scikit-learn matplotlib tqdm
```

### 3. Download the dataset
```python
import wfdb, os

DATA_DIR = './apnea_ecg_data'
os.makedirs(DATA_DIR, exist_ok=True)

RECORD_NAMES = (
    [f'a{i:02d}' for i in range(1, 21)] +
    [f'b{i:02d}' for i in range(1, 6)]  +
    [f'c{i:02d}' for i in range(1, 11)]
)

wfdb.dl_database('apnea-ecg', dl_dir=DATA_DIR,
                 records=RECORD_NAMES, annotators=['apn'])
```

### 4. Run the notebook
Open `apnea_prediction.ipynb` and run all cells top to bottom.

---

## ❓ Frequently Asked Questions (FAQS)

**Q: Does this actually detect apnea or predict it?**

It predicts it. The model outputs a warning ~45 seconds before the apnea event occurs, not during it. This is the key difference from most existing systems.

---

**Q: If apnea labels are used in training, how does it predict without them at inference?**

During training, we use known apnea timestamps to identify what ECG looks like just before each event — creating "prior" labels. At inference time, the model recognizes those same ECG patterns from the signal alone, without needing any apnea label. Think of it like teaching someone to recognize dark clouds — they don't need to wait for rain to know it's coming.

---

**Q: Why use ECG instead of a breathing sensor?**

The original paper (Chen et al., 2023) this work is inspired by used piezoelectric breathing sensors. ECG is harder — it measures heart activity, not breathing directly. But ECG is far more practical: it's cheap, wearable, and already used in routine monitoring. Our model achieves comparable accuracy (~87% vs 85.9%) using only ECG.

---

**Q: Why does the plot show prior and normal ECG windows looking the same?**

Exactly — that's the whole point. Pre-apnea ECG changes are subtle: slightly irregular R-R intervals, small shifts in heart rate variability, minor baseline noise. A human cannot reliably spot them across a 60-second window. The CNN-Transformer learns these patterns from thousands of labeled examples.

---

**Q: Can this be used in a real hospital?**

As a research prototype — yes, architecturally it is production-ready. A live ECG feed can be buffered in 60-second windows and fed to the model in real time. However, clinical deployment requires validation on a much larger and more diverse patient population, plus regulatory approval (FDA etc.).

---

**Q: What is 5-fold cross-validation and why use it?**

Instead of one fixed train/test split, we train and test 5 times on different data splits and average the results. This gives a more reliable estimate of how well the model generalizes to new patients, especially important with only 35 patients available.

---

**Q: Why not use all the data including apnea windows for training?**

Apnea windows are excluded because the task is early prediction, not apnea detection. Training on apnea windows would teach the model "what does apnea look like" — not "what does pre-apnea look like." Including them would confuse the classifier and reduce sensitivity on the prior class.


## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Model training |
| wfdb | PhysioNet data loading |
| NumPy | Signal processing |
| scikit-learn | Metrics, cross-validation |
| Matplotlib | Visualizations |
| Google Colab | Training environment (free T4 GPU) |

---

##  References

> Chen, Y., Yang, S., Li, H., Wang, L., & Wang, B. (2023). Prediction of Sleep Apnea Events Using a CNN–Transformer Network and Contactless Breathing Vibration Signals. *Bioengineering*, 10(7), 746. https://doi.org/10.3390/bioengineering10070746

> Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. (2000). The Apnea-ECG Database. *Computers in Cardiology*, 27, 255–258.

---

## 👤 Author

**Priyanshu** — Research Intern, BIT Mesra, Ranchi
