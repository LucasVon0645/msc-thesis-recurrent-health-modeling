import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HospReadmDataset(Dataset):
    """
    Builds per-visit samples with:
        {
            "x_past":    FloatTensor [max_seq_len, D_long],  # past only, padded
            "x_current": FloatTensor [D_curr],               # current visit features
            "mask_past": BoolTensor  [max_seq_len],          # True for valid past steps
            "y":         FloatTensor [],                     # scalar
            "subject_id": <original dtype>,
            "seq_len":   int,                                # number of valid past steps (<= max_seq_len)
            "t_index":   int,                                # 1-based index of current visit within subject
        }

    Notes
    -----
    - Past sequence includes ONLY visits strictly before the current one.
    - If reverse_chronological_order=True, the most recent past step is at index 0 of x_past.
      Padding follows the valid steps (left-aligned), i.e., x_past[:seq_len] are valid rows.
    """

    def __init__(
        self,
        csv_path: str,
        max_seq_len: int,
        longitudinal_feat_cols: list[str],     # features for x_past (sequence features)
        current_feat_cols: list[str],          # features for x_current (current-visit features)
        no_elective: bool = True,
        subject_id_col: str = "SUBJECT_ID",
        order_col: str = "ADMITTIME",
        label_col: str = "READMISSION_30_DAYS",
        next_admt_type_col: str = "NEXT_ADMISSION_TYPE",
        hosp_id_col: str = "HADM_ID",
        reverse_chronological_order: bool = True,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.max_seq_len = max_seq_len

        self.longitudinal_feat_cols = longitudinal_feat_cols
        self.current_feat_cols = current_feat_cols

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

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = []
        required = [self.subject_id_col, self.order_col, self.label_col]
        for col in required:
            if col not in df.columns:
                missing.append(col)
        for col in self.longitudinal_feat_cols:
            if col not in df.columns:
                missing.append(col)
        for col in self.current_feat_cols:
            if col not in df.columns:
                missing.append(col)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(set(missing))}")

    def _build_sequences(self) -> list[dict]:
        df = self._load_dataframe()
        self._validate_columns(df)

        # sort by time so rows are oldest→newest inside each subject
        df_sorted = df.sort_values([self.subject_id_col, self.order_col]).reset_index(drop=True)

        samples: list[dict] = []
        for subject_id, g in df_sorted.groupby(self.subject_id_col):
            g_long = g[self.longitudinal_feat_cols]
            g_curr = g[self.current_feat_cols]
            g_lab  = g[self.label_col]
            admit_types = g[self.next_admt_type_col] if self.next_admt_type_col in g.columns else None

            n = len(g)
            # Iterate t=1..n, where t indexes the *current* visit within the subject
            for t in range(1, n + 1):
                # label at current visit
                y_t = g_lab.iloc[t - 1]
                if pd.isna(y_t):
                    continue
                if self.no_elective and admit_types is not None and admit_types.iloc[t - 1] == "ELECTIVE":
                    continue

                # -----------------------
                # Build PAST sequence (strictly before current t)
                # -----------------------
                past = g_long.iloc[: t - 1].to_numpy(dtype=np.float32)  # shape [t-1, D_long]
                seq_len = past.shape[0]

                # keep the most recent 'max_seq_len' past visits
                if seq_len > self.max_seq_len:
                    past = past[-self.max_seq_len :, :]
                    seq_len = self.max_seq_len

                if self.reverse_chronological_order:
                    # most recent past at index 0
                    past = past[::-1]

                D_long = len(self.longitudinal_feat_cols)
                x_past = np.zeros((self.max_seq_len, D_long), dtype=np.float32)
                if seq_len > 0:
                    x_past[:seq_len, :] = past  # left-align valid steps

                mask_past = np.zeros((self.max_seq_len,), dtype=bool)
                if seq_len > 0:
                    mask_past[:seq_len] = True

                # -----------------------
                # Build CURRENT vector (at t)
                # -----------------------
                x_current = g_curr.iloc[t - 1].to_numpy(dtype=np.float32)  # shape [D_curr]

                samples.append(
                    {
                        "x_past": x_past,                       # np.float32 [max_seq_len, D_long]
                        "x_current": x_current,                 # np.float32 [D_curr]
                        "mask_past": mask_past,                 # bool [max_seq_len]
                        "y": float(y_t),
                        "subject_id": subject_id,
                        "seq_len": int(seq_len),                # number of valid past steps
                        "t_index": int(t),                      # 1-based within-subject index
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_current = torch.from_numpy(s["x_current"]).float()          # [D_curr]
        x_past = torch.from_numpy(s["x_past"]).float()                # [max_seq_len, D_long]
        mask_past = torch.from_numpy(s["mask_past"]).bool()              # [max_seq_len]
        y = torch.tensor(s["y"]).float()                             # scalar
        
        return x_current, x_past, mask_past, y