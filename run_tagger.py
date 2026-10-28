"""
run_tagger.py
Inference script for the trained BiLSTM + CharCNN POS tagger.
Usage: python3 run_tagger.py <input.sents> <output.out>
Loads tagger_checkpoint.pt from CWD and writes tagged output to CWD.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_IDX  = 0
UNK_IDX  = 1
CPAD_IDX = 0

MAX_CHAR_LEN = 30
CHAR_WIDTHS  = (2, 3, 4)  # must match training config


# ─────────────────────────────────────────────────────────────────────────────
#  Model definitions (must exactly match build_tagger.py)
# ─────────────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    def __init__(self, n_chars, char_dim, n_filters, widths, dropout=0.0, pad_idx=0):
        super().__init__()
        self.pad_idx  = pad_idx
        self.widths   = widths
        self.emb      = nn.Embedding(n_chars, char_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.convs    = nn.ModuleList(
            [nn.Conv1d(char_dim, n_filters, k) for k in widths]
        )
        self.out_dim  = n_filters * len(widths)

    def forward(self, chars):
        B, T, L = chars.shape
        x    = chars.reshape(B * T, L)
        mask = (x != self.pad_idx)
        e    = self.emb_drop(self.emb(x)).transpose(1, 2)
        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            z = F.relu(conv(e))
            valid = mask[:, k - 1:].unsqueeze(1)
            z = z.masked_fill(~valid, 0.0)
            feats.append(z.max(dim=2).values)
        c = torch.cat(feats, dim=1)
        return c.reshape(B, T, self.out_dim)


class BiLSTMCharCNNTagger(nn.Module):
    def __init__(self, vocab_size, n_chars, n_tags,
                 word_emb_dim, char_emb_dim, char_filters, char_widths,
                 hidden, n_layers, dropout=0.0, word_pad_idx=0, char_pad_idx=0):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.char_cnn = CharCNN(n_chars, char_emb_dim, char_filters,
                                char_widths, dropout=dropout, pad_idx=char_pad_idx)
        lstm_in = word_emb_dim + self.char_cnn.out_dim
        self.lstm = nn.LSTM(lstm_in, hidden,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.out_drop = nn.Dropout(dropout)
        self.fc       = nn.Linear(2 * hidden, n_tags)

    def forward(self, word_ids, char_ids, lengths):
        w  = self.word_emb(word_ids)
        c  = self.char_cnn(char_ids)
        x  = self.emb_drop(torch.cat([w, c], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=word_ids.size(1))
        return self.fc(self.out_drop(out))


# ─────────────────────────────────────────────────────────────────────────────
#  Character encoding helper
# ─────────────────────────────────────────────────────────────────────────────

def word_to_char_ids(word, char2idx, max_len=MAX_CHAR_LEN):
    word  = word[:max_len]
    chars = ["<"] + list(word) + [">"]
    return [char2idx.get(c, char2idx["<cUNK>"]) for c in chars]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_tagger.py <input.sents> <output.out>")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    start_time  = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load single checkpoint from CWD
    ckpt    = torch.load("tagger_checkpoint.pt", map_location=device)
    word2idx = ckpt["word2idx"]
    char2idx = ckpt["char2idx"]
    idx2tag  = {int(k): v for k, v in ckpt["idx2tag"].items()}
    cfg      = ckpt["config"]

    char_widths = tuple(cfg["char_widths"])

    model = BiLSTMCharCNNTagger(
        vocab_size   = cfg["vocab_size"],
        n_chars      = cfg["n_chars"],
        n_tags       = cfg["n_tags"],
        word_emb_dim = cfg["word_emb_dim"],
        char_emb_dim = cfg["char_emb_dim"],
        char_filters = cfg["char_filters"],
        char_widths  = char_widths,
        hidden       = cfg["hidden_dim"],
        n_layers     = cfg["num_layers"],
        dropout      = 0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(input_file, "r", encoding="utf-8") as f:
        raw_sentences = f.readlines()

    output_lines = []
    with torch.no_grad():
        for raw in raw_sentences:
            raw = raw.strip()
            if not raw:
                output_lines.append("")
                continue

            tokens   = raw.split()
            word_ids = [word2idx.get(tok, UNK_IDX) for tok in tokens]

            # Build char tensor (1, T, L)
            char_ids_list = [word_to_char_ids(tok, char2idx) for tok in tokens]
            T  = len(tokens)
            L  = max(len(c) for c in char_ids_list)
            L  = max(L, max(CHAR_WIDTHS))
            char_tensor = torch.zeros(1, T, L, dtype=torch.long, device=device)
            for t, ch in enumerate(char_ids_list):
                n = len(ch)
                char_tensor[0, t, :n] = torch.tensor(ch, dtype=torch.long)

            ids_t  = torch.tensor([word_ids], dtype=torch.long, device=device)
            lens_t = torch.tensor([T], dtype=torch.long)

            logits    = model(ids_t, char_tensor, lens_t)  # (1, T, K)
            pred_ids  = logits.argmax(-1)[0]               # (T,)
            pred_tags = [idx2tag[i.item()] for i in pred_ids]

            output_lines.append(" ".join(f"{tok}/{tag}"
                                         for tok, tag in zip(tokens, pred_tags)))

    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    elapsed = time.time() - start_time
    print(f"Inference complete: {len(output_lines)} sentences in {elapsed:.2f}s -> {output_file}")


if __name__ == "__main__":
    main()
