# ##########################################################################
# # Example of submission files
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├─ submission.py
# ├─ weights_challenge_1.pt
# └─ weights_challenge_2.pt

from braindecode.models import EEGNetv4
import torch


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = EEGNetv4(
            in_chans=129, n_classes=1, input_window_samples=int(2 * self.sfreq)
        ).to(self.device)
        # checkpoint_1 = torch.load("weights_challenge_1.pt", map_location=self.device)
        # model_challenge1.load_state_dict(...)
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNetv4(
            in_chans=129, n_classes=1, input_window_samples=int(2 * self.sfreq)
        ).to(self.device)
        # checkpoint_2 = torch.load("weights_challenge_2.pt", map_location=self.device)
        # model_challenge2.load_state_dict(...)
        return model_challenge2


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
