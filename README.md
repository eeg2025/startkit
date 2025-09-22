

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
