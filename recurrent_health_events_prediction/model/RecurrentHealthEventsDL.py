import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from collections import OrderedDict

class GRUNet(nn.Module):
    """
    GRU-based model for predicting health events from longitudinal and current features.
    A GRU processes the longitudinal sequence, and its final hidden state is concatenated
    with the current features to make a prediction via a feedforward head.
    The output is a single logit for binary classification for each input in the batch.
    Longitudinal sequences are padded to the right, and a mask is provided to indicate valid steps.
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        input_size_curr: int,
        hidden_size_head: int,
        input_size_seq: int,
        hidden_size_seq: int,
        num_layers_seq: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.num_layers_seq = num_layers_seq

        # GRU over past sequence only
        self.gru = nn.GRU(
            input_size=input_size_seq,
            hidden_size=hidden_size_seq,
            num_layers=num_layers_seq,
            batch_first=True,
            dropout=0.0 if num_layers_seq == 1 else dropout,
        )

        # head over [summary_past || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size_seq + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return False
    
    def forward(self,
                x_current: torch.Tensor,
                x_past: torch.Tensor,
                mask_past: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_past:    [B, T, D_long] float32  (past only; padded at the end)
            mask_past: [B, T]         bool     (True for valid steps in x_past)
            x_current: [B, D_curr]    float32  (current visit features)
        returns:   logits [B]
        """
        assert x_past is not None and mask_past is not None and x_current is not None, \
            "Must provide x_past, mask_past, and x_current"

        device = x_past.device
        B, T, D_long = x_past.shape

        # ensure boolean mask
        mask_past = mask_past.bool()
        lengths = mask_past.sum(dim=1)  # [B], number of valid past steps

        # placeholder summary
        h_last_past = torch.zeros(B, self.hidden_size_seq, device=device)

        has_past = lengths > 0
        if has_past.any():
            x_sel = x_past[has_past]                       # [B_sel, T, D_long]
            len_sel = lengths[has_past].to(torch.int64)    # [B_sel]

            # pack and run GRU
            packed = pack_padded_sequence(
                x_sel, lengths=len_sel.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)          # h_n: [num_layers, B_sel, H]
            h_last = h_n[-1]                   # [B_sel, H]
            h_last_past[has_past] = h_last


        feats = torch.cat([h_last_past, x_current], dim=1)   # [B, H + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)  # [B]
        return logits

class AttentionPoolingNet(nn.Module):
    """
    Attention-pooling model for predicting health events from longitudinal and current features.

    Replaces the GRU with:
        h_k = W v_k
        s_k = a^T h_k + b
        alpha_k = softmax(s_k)  (masked over valid steps)
        z = sum_k alpha_k * h_k

    Then concatenates z with x_current and feeds through a small MLP head.
    The output is a single logit per example (binary classification).

    Args:
        input_size_curr:  D_curr, size of x_current.
        hidden_size_head: width of the hidden layer in the classifier head.
        input_size_seq:   D_long, size of each step in x_past.
        hidden_size_seq:  m, projection size used in attention (m < D_long is typical).
        num_layers_seq:   kept for interface compatibility; unused here.
        dropout:          dropout used in the classifier head.
    """
    def __init__(
        self,
        input_size_curr: int,
        hidden_size_head: int,
        input_size_seq: int,
        hidden_size_seq: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.droput = dropout

        # Linear projection: h_k = W v_k
        self.proj = nn.Linear(input_size_seq, hidden_size_seq, bias=False)

        # s_k = a^T h_k + b   (a in R^m, b scalar)
        self.attn_vec = nn.Parameter(torch.randn(hidden_size_seq) * 0.02)
        self.attn_bias = nn.Parameter(torch.zeros(()))

        # head over [z || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size_seq + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return True

    def _attend(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked attention weights over time steps.

        H:    [B_sel, T, m]
        mask: [B_sel, T]  (True for valid steps)
        returns alpha: [B_sel, T]
        """
        # scores s_k = a^T h_k + b
        s = H.matmul(self.attn_vec) + self.attn_bias
        s = s.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(s, dim=1)
        # corner case: sample with no valid steps -> all -inf -> NaNs after softmax
        # we replace NaNs (if any) by zeros; caller will handle z=0 for those.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return alpha

    def forward(
        self,
        x_current: torch.Tensor,
        x_past: torch.Tensor,
        mask_past: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_past:    [B, T, D_long]  (reverse chronological; padded at the END)
            mask_past: [B, T]          bool, True for valid steps
            x_current: [B, D_curr]
        returns:
            logits: [B]
        """
        assert x_past is not None and mask_past is not None and x_current is not None, \
            "Must provide x_past, mask_past, and x_current"

        device = x_past.device
        B, T, _ = x_past.shape
        mask_past = mask_past.bool()
        lengths = mask_past.sum(dim=1)  # [B]

        # default pooled representation for everyone
        z = torch.zeros(B, self.hidden_size_seq, device=device)

        has_past = lengths > 0
        if has_past.any():
            x_sel = x_past[has_past]             # [B_sel, T, D_long]
            m_sel = mask_past[has_past]          # [B_sel, T]

            # project: H = W x
            # proj() applies linear layer to last dim
            H = self.proj(x_sel)                 # [B_sel, T, m]

            # attention weights (masked)
            alpha = self._attend(H, m_sel)       # [B_sel, T]

            # pooled: z = sum_t alpha_t * h_t
            # alpha.unsqueeze(1): [B_sel, 1, T]
            # bmm -> multiplies each matrix alpha [1, T] in the batch by H [T, m]
            z_sel = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B_sel, m]
            z[has_past] = z_sel

        feats = torch.cat([z, x_current], dim=1)        # [B, m + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)  # [B]
        return logits, alpha
