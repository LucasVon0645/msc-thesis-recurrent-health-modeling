import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HospReadmDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        max_seq_len: int,
        feature_cols: list[str],
        no_elective: bool = True,
        subject_id_col: str = "SUBJECT_ID",
        order_col: str = "ADMITTIME",
        label_col: str = "READMISSION_30_DAYS",
        next_admt_type_col: str = "NEXT_ADMISSION_TYPE",
        hosp_id_col: str = "HADM_ID",
        reverse_chronological_order: bool = True,  # NEW: most-recent-first if True
    ):
        super().__init__()
        self.csv_path = csv_path
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        self.no_elective = no_elective

        self.label_col = label_col
        self.subject_id_col = subject_id_col
        self.order_col = order_col
        self.hosp_id_col = hosp_id_col
        self.next_admt_type_col = next_admt_type_col
        self.reverse_chronological_order = reverse_chronological_order

        self.samples = self._build_sequences()

    def _load_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def _build_sequences(self) -> list[dict]:
        df = self._load_dataframe()

        if self.subject_id_col not in df.columns:
            raise ValueError(f"Subject id column '{self.subject_id_col}' not found in CSV.")
        if self.order_col not in df.columns:
            raise ValueError("Could not infer an order column; please pass order_col explicitly.")

        # sort by time so rows are oldest→newest inside each subject
        df_sorted = df.sort_values([self.subject_id_col, self.order_col]).reset_index(drop=True)

        samples = []
        for subject_id, g in df_sorted.groupby(self.subject_id_col):
            g_feat = g[self.feature_cols]
            g_lab  = g[self.label_col]
            admit_types = g[self.next_admt_type_col] if self.next_admt_type_col in g.columns else None

            n = len(g)
            for t in range(1, n + 1):
                y_t = g_lab.iloc[t - 1]
                if pd.isna(y_t):
                    continue
                if self.no_elective and admit_types is not None and admit_types.iloc[t - 1] == "ELECTIVE":
                    continue

                # build prefix up to visit t (oldest→newest)
                seq = g_feat.iloc[:t].to_numpy(dtype=np.float32)  # [t, d]
                seq_len, d = seq.shape

                # keep the most recent max_seq_len visits
                if seq_len > self.max_seq_len:
                    seq = seq[-self.max_seq_len:, :]
                    seq_len = self.max_seq_len

                # orientation: if True, flip so most recent is at index 0
                if self.reverse_chronological_order:
                    seq = seq[::-1]

                # pad to [max_seq_len, d] and build mask
                x = np.zeros((self.max_seq_len, d), dtype=np.float32)
                x[:seq_len, :] = seq  # left-align valid steps
                mask = np.zeros((self.max_seq_len,), dtype=bool)
                mask[:seq_len] = True

                samples.append({
                    "x": x,
                    "mask": mask,
                    "y": float(y_t),
                    "subject_id": subject_id,
                    "seq_len": int(seq_len),
                    "t_index": int(t),
                })

        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample["x"], dtype=torch.float32)          # [max_seq_len, d]
        mask = torch.tensor(sample["mask"], dtype=torch.bool)       # [max_seq_len]
        y = torch.tensor(sample["y"], dtype=torch.float32)          # scalar

        return x, mask, y
