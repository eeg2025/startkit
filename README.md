

# EEG Foundation Challenge Start Kits

This repository contains start kits for the [EEG Foundation challenges](https://eeg2025.github.io), a NeurIPS 2025 competition focused on advancing EEG decoding through cross-task transfer learning and psychopathology prediction.

## 🚀 Quick Start

### Challenge 1: Cross-Task Transfer Learning
<a target="_blank" href="https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Challenge 1 start-kit"/>
</a>

**Goal:** Develop models that can effectively transfer knowledge from passive EEG tasks to active cognitive tasks.

### Challenge 2: Predicting the P-factor from EEG
<a target="_blank" href="https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Challenge 2 start-kit"/>
</a>

**Goal:** Predict the psychopathology factor (P-factor) from EEG recordings to enable objective mental health assessments.

## 📁 Repository Structure

### Main Files

- **`challenge_1.ipynb`** - Complete tutorial for Challenge 1: Cross-task transfer learning
  - Understanding the Contrast Change Detection (CCD) task
  - Loading and preprocessing EEG data using EEGDash
  - Building deep learning models with Braindecode
  - Training and evaluation pipeline

- **`challenge_1.py`** - Python script version of Challenge 1 notebook for easier integration

- **`challenge_2.ipynb`** - Tutorial for Challenge 2: P-factor regression
  - Understanding the P-factor regression task
  - Data loading and windowing strategies
  - Model training for psychopathology prediction

- **`challenge_2.py`** - Python script version of Challenge 2 notebook for easier integration

- **`submission.py`** - Template for competition submission
  - Shows required format for model submission
  - Includes examples for both challenges

- **`requirements.txt`** - Python dependencies needed to run the notebooks

### Advanced Examples (not_ready_yet/)

- **`challenge_2_self_supervised.ipynb`** - Advanced self-supervised learning approach
  - Implementing Relative Positioning (RP) for unsupervised representation learning
  - Fine-tuning for P-factor prediction
  - PyTorch Lightning integration
  - *Note: This is an advanced example that may require additional setup*

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Main dependencies:
- `braindecode` - Deep learning library for EEG
- `eegdash` - Dataset management and preprocessing
- `pytorch` - Deep learning framework

## 🤝 Community & Support

This is a community competition with a strong open-source foundation. If you see something that doesn't work or could be improved:

1. **Please be kind** - we're all working together
2. Open an issue in the [issues tab](https://github.com/eeg2025/startkit/issues)
3. Join our weekly support sessions (starting 08/09/2025)

The entire decoding community will only go further when we stop solving the same problems over and over again, and start working together!


## 📚 Resources

- [Competition Website](https://eeg2025.github.io)
- [EEGDash Documentation](https://eeglab.org/EEGDash/overview.html)
- [Braindecode Models](https://braindecode.org/stable/models/models_table.html)
- [Dataset Download Guide](https://eeg2025.github.io/data/#downloading-the-data)


## Cluster dependencies

The dependencies that are available for inference are described below. If you need anything unavailable, we suggest you zip them together in the submission and put them inside your submission folder.


```bash
Package                      Version
---------------------------- ------------
acres                        0.5.0
aiobotocore                  2.24.2
aiohappyeyeballs             2.6.1
aiohttp                      3.12.15
aioitertools                 0.12.0
aiosignal                    1.4.0
async-timeout                5.0.1
attrs                        25.3.0
axial_positional_embedding   0.3.12
bids-validator               1.14.7.post0
bidsschematools              1.1.0
botocore                     1.40.18
braindecode                  1.2.0
certifi                      2025.8.3
cffi                         2.0.0
charset-normalizer           3.4.3
click                        8.2.1
CoLT5-attention              0.11.1
contourpy                    1.3.2
cycler                       0.12.1
decorator                    5.2.1
dnspython                    2.8.0
docopt                       0.6.2
docstring-inheritance        2.2.2
eegdash                      0.3.8
eeglabio                     0.1.0
einops                       0.8.1
filelock                     3.19.1
fonttools                    4.59.2
formulaic                    1.2.0
frozendict                   2.4.6
frozenlist                   1.7.0
fsspec                       2025.9.0
greenlet                     3.2.4
h5io                         0.2.5
h5py                         3.14.0
idna                         3.10
importlib_resources          6.5.2
interface-meta               1.3.0
Jinja2                       3.1.6
jmespath                     1.0.1
joblib                       1.5.2
kiwisolver                   1.4.9
lazy_loader                  0.4
lightning                    2.5.5
lightning-utilities          0.15.2
linear-attention-transformer 0.19.1
linformer                    0.2.3
llvmlite                     0.44.0
local-attention              1.10.0
MarkupSafe                   3.0.2
matplotlib                   3.10.6
mne                          1.10.1
mne-bids                     0.17.0
mpmath                       1.3.0
multidict                    6.6.4
narwhals                     2.4.0
networkx                     3.4.2
nibabel                      5.3.2
num2words                    0.5.14
numba                        0.61.2
numpy                        2.2.6
packaging                    25.0
pandas                       2.3.2
pillow                       11.3.0
pip                          25.2
platformdirs                 4.4.0
pooch                        1.8.2
product_key_memory           0.2.11
propcache                    0.3.2
pybids                       0.19.0
pycparser                    2.23
pymatreader                  1.1.0
pymongo                      4.15.0
pyparsing                    3.2.3
python-dateutil              2.9.0.post0
python-dotenv                1.1.1
pytorch-lightning            2.5.5
pytz                         2025.2
PyYAML                       6.0.2
requests                     2.32.5
s3fs                         2025.9.0
scikit-learn                 1.7.2
scipy                        1.15.3
setuptools                   78.1.1
six                          1.17.0
skorch                       1.2.0
soundfile                    0.13.1
SQLAlchemy                   2.0.43
sympy                        1.14.0
tabulate                     0.9.0
threadpoolctl                3.6.0
torch                        2.2.2
torchaudio                   2.2.2
torchinfo                    1.8.0
torchmetrics                 1.8.2
tqdm                         4.67.1
typing_extensions            4.15.0
tzdata                       2025.2
universal_pathlib            0.2.6
urllib3                      2.5.0
wfdb                         4.3.0
wheel                        0.45.1
wrapt                        1.17.3
xarray                       2025.6.1
xmltodict                    0.15.1
yarl                         1.20.1
```