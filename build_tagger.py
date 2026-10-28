"""
build_tagger.py
Training script for a BiLSTM + CharCNN POS tagger (Penn Treebank tag set).
Usage: python3 build_tagger.py train.tags
Saves a single checkpoint (tagger_checkpoint.pt) plus JSON vocab files to CWD.

Dimension note: word embedding (64) + CharCNN output (12 filters × 3 widths = 36)
= 100 total input dimensions to the BiLSTM — satisfying the ≤100 constraint.
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# *** Dimension constraint: word_emb (64) + char_cnn (12×3=36) = 100 ≤ 100 ***
WORD_EMB_DIM  = 64    # word embedding dimension
CHAR_EMB_DIM  = 32    # internal character embedding dimension
CHAR_FILTERS  = 12    # filters per CNN width  (12 × 3 widths = 36 char dims)
CHAR_WIDTHS   = (2, 3, 4)  # CNN kernel widths
# BiLSTM input = WORD_EMB_DIM + CHAR_FILTERS*len(CHAR_WIDTHS) = 64 + 36 = 100

HIDDEN_DIM    = 256   # hidden units per direction in BiLSTM
NUM_LAYERS    = 2     # number of stacked BiLSTM layers
DROPOUT       = 0.4   # dropout probability
BATCH_SIZE    = 64
MAX_EPOCHS    = 30
LR            = 1e-3
LR_PATIENCE   = 3     # epochs of stagnation before halving LR
GRAD_CLIP     = 5.0
FREQ_THRESH   = 1     # words with freq ≤ this mapped to UNK in word vocab
WORD_DROP_A   = 0.25  # word-dropout alpha (Kiperwasser & Goldberg, 2016)
MAX_CHAR_LEN  = 30    # truncate very long tokens for the char CNN

# ─── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1
CPAD_IDX  = 0     # character pad index
IGNORE    = -100  # ignored by CrossEntropyLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_tagged_file(path):
    """Read word/TAG lines into list of (words, tags) tuples.
    Uses rsplit('/', 1) to handle words that themselves contain slashes
    (e.g. '1/2/CD', 'and/or/CC').
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs = line.split()
            words, tags = [], []
            for pair in pairs:
                parts = pair.rsplit("/", 1)
                if len(parts) == 2:
                    words.append(parts[0])
                    tags.append(parts[1])
            if words:
                sentences.append((words, tags))
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
#  Vocabulary construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(sentences, freq_thresh=FREQ_THRESH):
    """Build word, char, and tag vocabularies from training sentences."""
    word_counts = Counter()
    char_set    = set()
    tag_set     = set()

    for words, tags in sentences:
        word_counts.update(words)
        tag_set.update(tags)
        for w in words:
            char_set.update(w)   # keep original case for char vocab

    # Word vocab: PAD=0, UNK=1, then frequent words (lowercased lookup at runtime)
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, cnt in word_counts.items():
        if cnt > freq_thresh:
            word2idx[word] = len(word2idx)

    # Char vocab: cPAD=0, cUNK=1, boundary markers <>=2,3, then all seen chars
    char2idx = {"<cPAD>": 0, "<cUNK>": 1, "<": 2, ">": 3}
    for ch in sorted(char_set):
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)

    # Tag vocab (sorted for determinism)
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    # Word frequency table keyed by word_idx (for word dropout)
    word_freq = {}
    for word, cnt in word_counts.items():
        wid = word2idx.get(word, UNK_IDX)
        word_freq[wid] = word_freq.get(wid, 0) + cnt

    return word2idx, char2idx, tag2idx, idx2tag, word_freq


def word_to_char_ids(word, char2idx, max_len=MAX_CHAR_LEN):
    """Convert a word to char indices with <word> boundary markers.
    Original case is preserved so the CharCNN can see capitalisation cues.
    """
    word = word[:max_len]
    chars = ["<"] + list(word) + [">"]
    return [char2idx.get(c, char2idx["<cUNK>"]) for c in chars]


def apply_word_dropout(word_id, word_freq, alpha=WORD_DROP_A):
    """Replace word_id with UNK with probability alpha/(alpha + count(word)).
    Characters are NOT dropped — the CharCNN must learn from spelling alone
    when the word embedding is masked to <UNK>.
    """
    if word_id <= UNK_IDX:
        return word_id
    count = word_freq.get(word_id, 1)
    p = alpha / (alpha + count)
    return UNK_IDX if random.random() < p else word_id


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class POSDataset(Dataset):
    def __init__(self, sentences, word2idx, char2idx, tag2idx,
                 word_freq=None, train_mode=False, word_dropout_alpha=None,
                 use_charcnn=True):
        self.data       = sentences
        self.word2idx   = word2idx
        self.char2idx   = char2idx
        self.tag2idx    = tag2idx
        self.word_freq  = word_freq
        self.train_mode = train_mode
        self.use_charcnn = use_charcnn
        if word_dropout_alpha is None:
            self.word_dropout_alpha = WORD_DROP_A
        else:
            self.word_dropout_alpha = word_dropout_alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]

        # Word indices (original case preserved in vocab; lowercase fallback below)
        word_ids = [self.word2idx.get(w, UNK_IDX) for w in words]
        tag_ids  = [self.tag2idx[t] for t in tags]

        # Word dropout (training only) — character sequences untouched
        if self.train_mode and self.word_freq is not None and self.word_dropout_alpha and self.word_dropout_alpha > 0:
            word_ids = [apply_word_dropout(wid, self.word_freq, self.word_dropout_alpha) for wid in word_ids]

        # Character indices — keep original case for capitalisation cue
        if self.use_charcnn:
            char_ids = [word_to_char_ids(w, self.char2idx) for w in words]
        else:
            char_ids = None

        return (
            torch.tensor(word_ids, dtype=torch.long),
            char_ids,   # list of variable-length lists (padded in collate_fn)
            torch.tensor(tag_ids,  dtype=torch.long),
        )


def collate_fn(batch):
    """Collate a list of (word_ids, char_ids, tag_ids) into padded tensors."""
    word_seqs, char_seqs_batch, tag_seqs = zip(*batch)

    lengths  = torch.tensor([len(s) for s in word_seqs], dtype=torch.long)
    word_pad = pad_sequence(word_seqs, batch_first=True, padding_value=PAD_IDX)
    # Pad targets with IGNORE=-100, NOT 0 (which is a real tag index!)
    tag_pad  = pad_sequence(tag_seqs,  batch_first=True, padding_value=IGNORE)

    if char_seqs_batch and char_seqs_batch[0] is None:
        char_pad = None
        return word_pad, char_pad, tag_pad, lengths

    B = len(word_seqs)
    T = word_pad.size(1)
    L = max(max(len(c) for c in sent_chars) for sent_chars in char_seqs_batch)
    L = max(L, max(CHAR_WIDTHS))  # must be >= widest CNN kernel

    # (B, T, L) character tensor padded with CPAD_IDX=0
    char_pad = torch.full((B, T, L), CPAD_IDX, dtype=torch.long)
    for b, sent_chars in enumerate(char_seqs_batch):
        for t, ch_ids in enumerate(sent_chars):
            n = len(ch_ids)
            char_pad[b, t, :n] = torch.tensor(ch_ids, dtype=torch.long)

    return word_pad, char_pad, tag_pad, lengths


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    """
    Character-level CNN encoder.

    For each word, it runs multiple Conv1d filters of different widths
    over the character embedding sequence, then applies max-over-time
    pooling to produce a fixed-size feature vector.

    Masking: windows that cover padding characters are set to 0.0 before
    the max, preventing pad-column bias from dominating short words.

    Input : chars (N, L)  — N words, L characters each (padded with CPAD)
    Output: (N, out_dim)  where out_dim = n_filters * len(widths)
    """

    def __init__(self, n_chars, char_dim=CHAR_EMB_DIM,
                 n_filters=CHAR_FILTERS, widths=CHAR_WIDTHS,
                 dropout=0.5, pad_idx=CPAD_IDX):
        super().__init__()
        self.pad_idx  = pad_idx
        self.widths   = widths
        self.emb      = nn.Embedding(n_chars, char_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.convs    = nn.ModuleList(
            [nn.Conv1d(char_dim, n_filters, k) for k in widths]
        )
        self.out_dim = n_filters * len(widths)  # 12 × 3 = 36

    def forward(self, chars):
        """
        chars: (B, T, L)
        returns: (B, T, out_dim)
        """
        B, T, L = chars.shape
        x    = chars.reshape(B * T, L)                     # (N, L)
        mask = (x != self.pad_idx)                          # (N, L) bool
        e    = self.emb_drop(self.emb(x)).transpose(1, 2)  # (N, d_c, L)

        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            z = F.relu(conv(e))                             # (N, F, L-k+1)
            # A window at position i covers chars i..i+k-1.
            # It is valid iff char i+k-1 is a real character (not cPAD).
            valid = mask[:, k - 1:].unsqueeze(1)            # (N, 1, L-k+1)
            # After ReLU all real outputs are ≥ 0, so masking with 0.0 is safe.
            z = z.masked_fill(~valid, 0.0)
            feats.append(z.max(dim=2).values)               # (N, F)

        c = torch.cat(feats, dim=1)                         # (N, out_dim=36)
        return c.reshape(B, T, self.out_dim)                # (B, T, 36)


class BiLSTMCharCNNTagger(nn.Module):
    """
    Bidirectional LSTM POS tagger with character-level CNN input features.

    Architecture (one position t):
      e_t  = WordEmb[x_t]               ∈ R^64     (word embedding)
      c_t  = CharCNN(chars of x_t)      ∈ R^36     (char features)
      x_t  = Dropout([e_t ; c_t])       ∈ R^100    (joint input)
      →h_t = LSTMfwd(x_t, →h_{t-1})    ∈ R^H
      ←h_t = LSTMbwd(x_t, ←h_{t+1})   ∈ R^H
      h_t  = [→h_t ; ←h_t]             ∈ R^{2H}
      z_t  = W · Dropout(h_t) + b       ∈ R^K
      ŷ_t  = argmax_k z_{t,k}           (predicted POS tag)
    """

    def __init__(self, vocab_size, n_chars, n_tags,
                 word_emb_dim=WORD_EMB_DIM,
                 char_emb_dim=CHAR_EMB_DIM,
                 char_filters=CHAR_FILTERS,
                 char_widths=CHAR_WIDTHS,
                 hidden=HIDDEN_DIM,
                 n_layers=NUM_LAYERS,
                 dropout=DROPOUT,
                 word_pad_idx=PAD_IDX,
                 char_pad_idx=CPAD_IDX):
        super().__init__()

        # Word embedding branch
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_pad_idx)
        self.emb_drop = nn.Dropout(dropout)

        # Character CNN branch
        self.char_cnn = CharCNN(n_chars, char_emb_dim, char_filters,
                                char_widths, dropout=dropout, pad_idx=char_pad_idx)

        # BiLSTM: input = 64 word + 36 char = 100 dims
        lstm_in = word_emb_dim + self.char_cnn.out_dim  # must be ≤ 100
        assert lstm_in <= 100, f"Input dim {lstm_in} exceeds 100!"

        self.lstm = nn.LSTM(lstm_in, hidden,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        self.out_drop = nn.Dropout(dropout)
        self.fc       = nn.Linear(2 * hidden, n_tags)

    def forward(self, word_ids, char_ids, lengths):
        """
        word_ids : (B, T)       — padded word indices
        char_ids : (B, T, L)    — padded character indices (original case)
        lengths  : (B,)         — true sentence lengths (for packing)
        returns  : (B, T, K)    — logits over K POS tags
        """
        w  = self.word_emb(word_ids)      # (B, T, 64)
        c  = self.char_cnn(char_ids)      # (B, T, 36)
        x  = self.emb_drop(torch.cat([w, c], dim=-1))  # (B, T, 100)

        # Pack so backward LSTM never reads <PAD> tokens
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=word_ids.size(1))  # (B, T, 2H)

        return self.fc(self.out_drop(out))  # (B, T, K)


# ─────────────────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch=1, max_epochs=MAX_EPOCHS):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{max_epochs} [Train]", unit="batch", leave=False)
    for word_ids, char_ids, tag_ids, lengths in pbar:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        tag_ids  = tag_ids.to(device)

        logits  = model(word_ids, char_ids, lengths)   # (B, T, K)
        B, T, K = logits.shape
        loss    = criterion(logits.reshape(B * T, K), tag_ids.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        pred  = logits.argmax(-1)
        mask  = tag_ids != IGNORE
        correct += (pred[mask] == tag_ids[mask]).sum().item()
        total   += mask.sum().item()

        running_loss = total_loss / (pbar.n + 1)
        running_acc  = (correct / total * 100) if total > 0 else 0.0
        pbar.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.2f}%"})

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for word_ids, char_ids, tag_ids, lengths in pbar:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        tag_ids  = tag_ids.to(device)

        logits  = model(word_ids, char_ids, lengths)
        B, T, K = logits.shape
        loss    = criterion(logits.reshape(B * T, K), tag_ids.reshape(B * T))

        total_loss += loss.item()
        pred  = logits.argmax(-1)
        mask  = tag_ids != IGNORE
        correct += (pred[mask] == tag_ids[mask]).sum().item()
        total   += mask.sum().item()

    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 build_tagger.py train.tags [dev.tags]")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file  = sys.argv[2] if len(sys.argv) > 2 else ("dev.tags" if os.path.exists("dev.tags") else None)
    start_time = time.time()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # ── Load & split data ─────────────────────────────────────────────────
    print("Loading training data ...", flush=True)
    all_sentences = parse_tagged_file(train_file)
    print(f"  {len(all_sentences)} sentences loaded from {train_file}.", flush=True)

    random.shuffle(all_sentences)
    split       = int(0.90 * len(all_sentences))   # 90% train / 10% internal val
    train_sents = all_sentences[:split]
    val_sents   = all_sentences[split:]
    print(f"  Train: {len(train_sents)} | Val: {len(val_sents)}", flush=True)

    test_sents = parse_tagged_file(test_file) if test_file and os.path.exists(test_file) else None
    if test_sents:
        print(f"  Test ({test_file}): {len(test_sents)} sentences loaded.", flush=True)

    # ── Build vocabularies ────────────────────────────────────────────────
    print("Building vocabularies ...", flush=True)
    word2idx, char2idx, tag2idx, idx2tag, word_freq = build_vocab(
        train_sents, FREQ_THRESH
    )
    vocab_size = len(word2idx)
    n_chars    = len(char2idx)
    n_tags     = len(tag2idx)
    char_dim_out = CHAR_FILTERS * len(CHAR_WIDTHS)
    print(f"  Words: {vocab_size} | Chars: {n_chars} | Tags: {n_tags}", flush=True)
    print(f"  Input dim to BiLSTM: {WORD_EMB_DIM} (word) + {char_dim_out} (char) = "
          f"{WORD_EMB_DIM + char_dim_out} (must be <= 100)", flush=True)

    # ── Save all artifacts to CWD in a single checkpoint ─────────────────
    config = {
        "word_emb_dim": WORD_EMB_DIM,
        "char_emb_dim": CHAR_EMB_DIM,
        "char_filters": CHAR_FILTERS,
        "char_widths":  list(CHAR_WIDTHS),
        "hidden_dim":   HIDDEN_DIM,
        "num_layers":   NUM_LAYERS,
        "dropout":      DROPOUT,
        "vocab_size":   vocab_size,
        "n_chars":      n_chars,
        "n_tags":       n_tags,
    }
    # Also save JSON vocab files for inspection
    with open("word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    with open("char2idx.json", "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False)
    with open("idx2tag.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2tag.items()}, f, ensure_ascii=False)
    with open("config.json", "w") as f:
        json.dump(config, f)

    # ── Build datasets & dataloaders ──────────────────────────────────────
    train_ds = POSDataset(train_sents, word2idx, char2idx, tag2idx,
                          word_freq=word_freq, train_mode=True)
    val_ds   = POSDataset(val_sents,   word2idx, char2idx, tag2idx,
                          train_mode=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    if test_sents:
        test_ds = POSDataset(test_sents, word2idx, char2idx, tag2idx, train_mode=False)
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
    else:
        test_loader = None

    # ── Model ─────────────────────────────────────────────────────────────
    model = BiLSTMCharCNNTagger(
        vocab_size=vocab_size, n_chars=n_chars, n_tags=n_tags
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}", flush=True)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_epoch    = 0

    # Curve tracking for report figures
    curve_train_losses = []
    curve_train_accs   = []
    curve_val_losses   = []
    curve_val_accs     = []
    curve_test_accs    = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                      epoch=epoch, max_epochs=MAX_EPOCHS)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device,
                                   desc=f"Epoch {epoch:02d}/{MAX_EPOCHS} [Val]")
        scheduler.step(vl_acc)

        ts_acc = None
        if test_loader is not None:
            _, ts_acc = evaluate(model, test_loader, criterion, device,
                                 desc=f"Epoch {epoch:02d}/{MAX_EPOCHS} [Test]")

        elapsed = time.time() - t0

        # Record curves
        curve_train_losses.append(round(tr_loss, 6))
        curve_train_accs.append(round(tr_acc * 100, 4))
        curve_val_losses.append(round(vl_loss, 6))
        curve_val_accs.append(round(vl_acc * 100, 4))
        curve_test_accs.append(round(ts_acc * 100, 4) if ts_acc is not None else None)

        if ts_acc is not None:
            print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}% | "
                  f"Test Acc: {ts_acc*100:.2f}% | "
                  f"Time: {elapsed:.1f}s", flush=True)
        else:
            print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}% | "
                  f"Time: {elapsed:.1f}s", flush=True)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            if ts_acc is not None:
                best_test_acc = ts_acc
            # Save a single checkpoint with model + all vocab dicts
            torch.save({
                "model_state": model.state_dict(),
                "word2idx":    word2idx,
                "char2idx":    char2idx,
                "idx2tag":     idx2tag,
                "config":      config,
            }, "tagger_checkpoint.pt")
            if ts_acc is not None:
                print(f"  >> Best model saved (val acc = {best_val_acc*100:.2f}%, test acc = {best_test_acc*100:.2f}%)", flush=True)
            else:
                print(f"  >> Best model saved (val acc = {best_val_acc*100:.2f}%)", flush=True)

        # Save curves after every epoch so they survive interruptions
        curves = {
            "train_losses": curve_train_losses,
            "train_accs":   curve_train_accs,
            "val_losses":   curve_val_losses,
            "val_accs":     curve_val_accs,
            "test_accs":    curve_test_accs,
            "best_epoch":   best_epoch,
            "best_val_acc": round(best_val_acc * 100, 4),
            "best_test_acc": round(best_test_acc * 100, 4),
        }
        with open("full_model_curves.json", "w") as cf:
            json.dump(curves, cf, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} min.", flush=True)
    if best_test_acc > 0:
        print(f"Best model: epoch {best_epoch}, val acc = {best_val_acc*100:.2f}%, test acc = {best_test_acc*100:.2f}%", flush=True)
    else:
        print(f"Best model: epoch {best_epoch}, val acc = {best_val_acc*100:.2f}%", flush=True)

    # Final curve summary
    print(f"Training curves saved to full_model_curves.json", flush=True)


if __name__ == "__main__":
    main()
