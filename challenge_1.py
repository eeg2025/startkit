# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Challenge 1: Cross-Task Transfer Learning!
#
# ## How can we use the knowledge from one EEG Decoding task into another?
#
# Transfer learning is a widespread technique used in deep learning. It uses knowledge learned from one source task/domain in another target task/domain. It has been studied in depth in computer vision, natural language processing, and speech, but what about EEG brain decoding?
#
# The cross-task transfer learning scenario in EEG decoding is remarkably underexplored in comparison to the developers of new models, [Aristimunha et al., (2023)](https://arxiv.org/abs/2308.02408), even though it can be much more useful for real applications, see [Wimpff et al. (2025)](https://arxiv.org/abs/2502.06828), [Wu et al. (2025)](https://arxiv.org/abs/2507.09882).
#
# Our Challenge 1 addresses a key goal in neurotechnology: decoding cognitive function from EEG using the pre-trained knowledge from another. In other words, developing models that can effectively transfer/adapt/adjust/fine-tune knowledge from passive EEG tasks to active tasks.
#
# The ability to generalize and transfer is something critical that we believe should be focused. To go beyond just comparing metrics numbers that are often not comparable, given the specificities of EEG, such as pre-processing, inter-subject variability, and many other unique components of this type of data.
#
# This means your submitted model might be trained on a subset of tasks and fine-tuned on data from another condition, evaluating its capacity to generalize with task-specific fine-tuning.

# %% [markdown]
# __________
#
# Note: For simplicity purposes, we will only show how to do the decoding directly in our target task, and it is up to the teams to think about how to use the passive task to perform the pre-training.

# %% [markdown]
# ---
# ## Summary table for this start kit
#
# In this tutorial, we are going to show in more detail what we want from Challenge 1:
#
# **Contents**:
#
# 0. Understand the Contrast Change Detection - CCD task.
# 1. Understand the [`EEGChallengeDataset`](https://eeglab.org/EEGDash/api/eegdash.html#eegdash.EEGChallengeDataset) object.
# 2. Preparing the dataloaders.
# 3. Building the deep learning model with [`braindecode`](https://braindecode.org/stable/models/models_table.html).
# 4. Designing the training loop.
# 5. Training the model.
# 6. Evaluating test performance.
# 7. Going further, *benchmark go brrr!*

# %% Install dependencies on colab [code] tags=["hide-input"]
# !pip install braindecode
# !pip install eegdash

# %% Imports and setup [code]
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import mne
from mne_bids import get_bids_path_from_fname
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.models import EEGNeX
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from typing import Optional
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import copy
from joblib import Parallel, delayed

# Identify whether a CUDA-enabled GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    msg = 'CUDA-enabled GPU found. Training should be faster.'
else:
    msg = (
        "No GPU found. Training will be carried out on CPU, which might be "
        "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
        " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
        "selecting 'T4 GPU'\nunder 'Hardware accelerator'."
    )
print(msg)

# %% [markdown]
# ## 1. What are we decoding?
#
# The Contrast Change Detection (CCD) task relates to [Steady-State Visual Evoked Potentials (SSVEP)](https://en.wikipedia.org/wiki/Steady-state_visually_evoked_potential) and [Event-Related Potentials (ERP)](https://en.wikipedia.org/wiki/Event-related_potential).
#
# Algorithmically, what the subject sees during recording is:
#
# * Two flickering striped discs: one tilted left, one tilted right.
# * After a variable delay, **one disc's contrast gradually increases** **while the other decreases**.
# * They **press left or right** to indicate which disc got stronger.
# * They receive **feedback** (🙂 correct / 🙁 incorrect).
#
# **The task parallels SSVEP and ERP:**
#
# * The continuous flicker **tags the EEG at fixed frequencies (and harmonics)** → SSVEP-like signals.
# * The **ramp onset**, the **button press**, and the **feedback** are **time-locked events** that yield ERP-like components.
#
# Your task (**label**) is to predict the response time for the subject during this windows.

# %% Load the data [code]
from pathlib import Path
from eegdash.dataset import EEGChallengeDataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

dataset_ccd = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R5", 
    cache_dir=DATA_DIR,
    mini=True
)

# For visualization purposes, we will see just one object.
raw = dataset_ccd.datasets[0].raw  # get the Raw object of the first recording

# To download all data directly, you can do:
# raws = Parallel(n_jobs=-1)(
#     delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets
# )

# %% Define utility functions for epoching [code]
# Extract events:
def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast trial with stimulus/response metrics."""
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"])
        end = float(tr["next_onset"])

        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        if not np.isnan(stim_onset):
            resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
        else:
            resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

        if resp_block.empty:
            resp_onset = np.nan
            resp_type = None
            feedback = None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type = resp_block.iloc[0]["value"]
            feedback = resp_block.iloc[0]["feedback"]

        rt_from_stim = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
        rt_from_trial = (resp_onset - start) if not np.isnan(resp_onset) else np.nan

        correct = None
        if isinstance(feedback, str):
            if feedback == "smiley_face": correct = True
            elif feedback == "sad_face": correct = False

        rows.append({
            "trial_start_onset": start,
            "trial_stop_onset": end,
            "stimulus_onset": stim_onset,
            "response_onset": resp_onset,
            "rt_from_stimulus": rt_from_stim,
            "rt_from_trialstart": rt_from_trial,
            "response_type": resp_type,
            "correct": correct,
        })

    return pd.DataFrame(rows)

# Aux functions to inject the annot
def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)

def _to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None

def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0,
                                require_stimulus=True, require_response=True):
    """Create 'contrast_trial_start' annotations with float target in extras."""
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath

    events_df = (pd.read_csv(events_file, sep="\t")
                   .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
                   .sort_values("onset", kind="mergesort").reset_index(drop=True))

    trials = build_trial_table(events_df)

    if require_stimulus:
        trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:
        trials = trials[trials["response_onset"].notna()].copy()

    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)

    onsets = trials["trial_start_onset"].to_numpy(float)
    durations = np.full(len(trials), float(epoch_length), dtype=float)
    descs = ["contrast_trial_start"] * len(trials)

    extras = []
    for i, v in enumerate(targets):
        row = trials.iloc[i]

        extras.append({
            "target": _to_float_or_none(v),
            "rt_from_stimulus": _to_float_or_none(row["rt_from_stimulus"]),
            "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
            "stimulus_onset": _to_float_or_none(row["stimulus_onset"]),
            "response_onset": _to_float_or_none(row["response_onset"]),
            "correct": _to_int_or_none(row["correct"]),
            "response_type": _to_str_or_none(row["response_type"]),
        })

    new_ann = mne.Annotations(onset=onsets, duration=durations, description=descs,
                              orig_time=raw.info["meas_date"], extras=extras)
    raw.set_annotations(new_ann, verbose=False)
    return raw

def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann = raw.annotations
    mask = (ann.description == "contrast_trial_start")
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex["stimulus_onset"]
        resp_t = ex["response_onset"]

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex["rt_from_trialstart"]
            rts = ex["rt_from_stimulus"]
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex["rt_from_trialstart"]
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if (stim_t is not None) and not (isinstance(stim_t, float) and np.isnan(stim_t)):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if (resp_t is not None) and not (isinstance(resp_t, float) and np.isnan(resp_t)):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw

def add_extras_columns(
    windows_concat_ds,
    original_concat_ds,
    desc="contrast_trial_start",
    keys=("target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset","correct","response_type"),
):
    float_cols = {"target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset"}

    for win_ds, base_ds in zip(windows_concat_ds.datasets, original_concat_ds.datasets):
        ann = base_ds.raw.annotations
        idx = np.where(ann.description == desc)[0]
        if idx.size == 0:
            continue

        per_trial = [
            {k: (ann.extras[i][k] if ann.extras is not None and k in ann.extras[i] else None) for k in keys}
            for i in idx
        ]

        md = win_ds.metadata.copy()
        first = (md["i_window_in_trial"].to_numpy() == 0)
        trial_ids = first.cumsum() - 1
        n_trials = trial_ids.max() + 1 if len(trial_ids) else 0
        assert n_trials == len(per_trial), f"Trial mismatch: {n_trials} vs {len(per_trial)}"

        for k in keys:
            vals = [per_trial[t][k] if t < len(per_trial) else None for t in trial_ids]
            if k == "correct":
                ser = pd.Series([None if v is None else int(bool(v)) for v in vals],
                                index=md.index, dtype="Int64")
            elif k in float_cols:
                ser = pd.Series([np.nan if v is None else float(v) for v in vals],
                                index=md.index, dtype="Float64")
            else:  # response_type
                ser = pd.Series(vals, index=md.index, dtype="string")

            # Replace the whole column to avoid dtype conflicts
            md[k] = ser

        win_ds.metadata = md.reset_index(drop=True)
        if hasattr(win_ds, "y"):
            y_np = win_ds.metadata["target"].astype(float).to_numpy()
            win_ds.y = y_np[:, None]  # (N, 1)

    return windows_concat_ds

def keep_only_recordings_with(desc, concat_ds):
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            print(f"[warn] Recording {ds.raw.filenames[0]} does not contain event '{desc}'")
    return BaseConcatDataset(kept)

# %% Create windows of interest [code]
EPOCH_LEN_S = 2.0
SFREQ = 100  # by definition here

transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
        require_stimulus=True, require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd, transformation_offline, n_jobs=1)

ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

# Keep only recordings that actually contain stimulus anchors
dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

# Create single-interval windows (stim-locked, long enough to include the response)
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),                 # +0.5 s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),   # +2.5 s
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)

# Injecting metadata into the extra mne annotation.
single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
)

# %% Split the data [code]
# Extract meta information
meta_information = single_windows.get_metadata()

valid_frac = 0.1
test_frac = 0.1
seed = 2025

# List of subjects to remove due to quality issues (insufficient channels or duration)
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]

subjects = meta_information["subject"].unique()
# Filter out problematic subjects
subjects = [s for s in subjects if s not in sub_rm]

train_subj, valid_test_subject = train_test_split(
    subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True
)

valid_subj, test_subj = train_test_split(
    valid_test_subject, test_size=test_frac, random_state=check_random_state(seed + 1), shuffle=True
)

# Sanity check
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

# Create train/valid/test splits for the windows
subject_split = single_windows.split("subject")
train_set = []
valid_set = []
test_set = []

for s in subject_split:
    if s in train_subj:
        train_set.append(subject_split[s])
    elif s in valid_subj:
        valid_set.append(subject_split[s])
    elif s in test_subj:
        test_set.append(subject_split[s])

train_set = BaseConcatDataset(train_set)
valid_set = BaseConcatDataset(valid_set)
test_set = BaseConcatDataset(test_set)

print("Number of examples in each split in the minirelease")
print(f"Train:\t{len(train_set)}")
print(f"Valid:\t{len(valid_set)}")
print(f"Test:\t{len(test_set)}")

# %% Create dataloaders [code]
batch_size = 128
num_workers = 1 # We are using a single worker, but you can increase this for faster data loading

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# %% Build the model [code]
# For any braindecode model, you can initialize only inputing the signal related parameters
# You can use any Pytorch module that you want here.
model = EEGNeX(
    n_chans=129,      # 129 channels
    n_outputs=1,      # 1 output for regression
    n_times=200,      # 2 seconds
    sfreq=100,        # sample frequency 100 Hz
)

print(model)
model.to(device)

# %% Define training functions [code]
def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
):
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        # Support datasets that may return (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Flatten to 1D for regression metrics and accumulate squared error
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse

@torch.no_grad()
def valid_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats
    )

    for batch_idx, batch in iterator:
        # Supports (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
    return avg_loss, rmse

# %% Train the model [code]
lr = 1E-3
weight_decay = 1E-5
n_epochs = 100
early_stopping_patience = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
loss_fn = torch.nn.MSELoss()

patience = 5
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, device
    )
    val_loss, val_rmse = valid_model(test_loader, model, loss_fn, device)

    print(
        f"Train RMSE: {train_rmse:.6f}, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Val RMSE: {val_rmse:.6f}, "
        f"Average Val Loss: {val_loss:.6f}"
    )

    if val_rmse < best_rmse - min_delta:
        best_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# %% Save the model [code]
torch.save(model.state_dict(), "weights_challenge_1.pt")
print("Model saved as 'weights_challenge_1.pt'")