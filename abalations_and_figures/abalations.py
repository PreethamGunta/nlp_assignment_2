"""
abalations.py
============
Merged ablation study for the Hybrid Char-CNN + Word-Embedding BiLSTM POS tagger.

Usage:
    python3 abalations.py data/train.tags data/dev.sents data/dev.tags [--epochs N]

Results saved to data/ablation_results.json
"""

import sys
import os
import json
import csv
import copy
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)
from build_tagger import (
    parse_tagged_file, build_vocab,
    POSDataset, collate_fn,
    CharCNN, word_to_char_ids,
    PAD_TOKEN, UNK_TOKEN,
    WORD_DROP_A as DEFAULT_ALPHA,
    IGNORE as IGNORE_IDX,
)

SEED = 42
def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[abalations] Device: {DEVICE}")

ABLATION_EPOCHS = 15
PATIENCE        = 3
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
MAX_GRAD_NORM   = 5.0
CHAR_EMB_DIM    = 30


@dataclass
class AblationConfig:
    name: str
    word_emb_dim: int = 64
    char_emb_dim: int = CHAR_EMB_DIM
    char_filters: int = 12
    char_widths: Tuple[int, ...] = (2, 3, 4)
    use_charcnn: bool = True
    hidden: int = 256
    n_layers: int = 2
    dropout: float = 0.5
    bidirectional: bool = True
    word_drop_alpha: float = DEFAULT_ALPHA


class AblationModel(nn.Module):
    def __init__(self, vocab_size, n_chars, n_tags,
                 word_emb_dim=64, char_emb_dim=CHAR_EMB_DIM, char_filters=12,
                 char_widths=(2, 3, 4), hidden=256, n_layers=2,
                 dropout=0.4, bidirectional=True, use_charcnn=True):
        super().__init__()
        self.use_charcnn = use_charcnn and char_filters > 0
        self.bidirectional = bidirectional
        if word_emb_dim and word_emb_dim > 0:
            self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        else:
            self.word_emb = None
        self.word_dim = word_emb_dim if (word_emb_dim and word_emb_dim > 0) else 0
        if self.use_charcnn:
            self.char_cnn = CharCNN(n_chars, char_emb_dim, char_filters,
                                    tuple(char_widths), dropout=dropout, pad_idx=0)
        else:
            self.char_cnn = None
        lstm_in = self.word_dim + (self.char_cnn.out_dim if self.char_cnn is not None else 0)
        if lstm_in == 0:
            lstm_in = 1
        dirs = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(lstm_in, hidden, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(dirs * hidden, n_tags)

    def forward(self, word_ids, char_ids, lengths):
        parts = []
        if self.word_emb is not None:
            parts.append(self.word_emb(word_ids))
        if self.char_cnn is not None and char_ids is not None:
            parts.append(self.char_cnn(char_ids))
        if len(parts) > 1:
            x = torch.cat(parts, dim=2)
        elif len(parts) == 1:
            x = parts[0]
        else:
            x = torch.zeros(word_ids.size(0), word_ids.size(1), 1, device=word_ids.device)
        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=word_ids.size(1))
        out = self.dropout(out)
        return self.fc(out)


def make_collate(char_widths, use_charcnn=True):
    def collate_fn(batch):
        word_seqs, char_seqs_batch, tag_seqs = zip(*batch)
        lengths  = torch.tensor([len(s) for s in word_seqs], dtype=torch.long)
        word_pad = torch.nn.utils.rnn.pad_sequence(word_seqs, batch_first=True, padding_value=0)
        tag_pad  = torch.nn.utils.rnn.pad_sequence(tag_seqs, batch_first=True, padding_value=IGNORE_IDX)
        if use_charcnn and char_seqs_batch and char_seqs_batch[0] is not None:
            B = len(word_seqs)
            T = word_pad.size(1)
            L = max(max(len(c) for c in sent_chars) for sent_chars in char_seqs_batch)
            L = max(L, max(char_widths))
            char_pad = torch.full((B, T, L), 0, dtype=torch.long)
            for b, sent_chars in enumerate(char_seqs_batch):
                for t, ch_ids in enumerate(sent_chars):
                    n = len(ch_ids)
                    char_pad[b, t, :n] = torch.tensor(ch_ids, dtype=torch.long)
        else:
            char_pad = None
        return word_pad, char_pad, tag_pad, lengths
    return collate_fn


def build_configs() -> List[AblationConfig]:
    configs = []
    configs.append(AblationConfig(name="char_only (0+99)", word_emb_dim=0,
        char_filters=33, char_widths=(2,3,4), use_charcnn=True, char_emb_dim=CHAR_EMB_DIM,
        word_drop_alpha=DEFAULT_ALPHA))
    configs.append(AblationConfig(name="word_only (100+0)", word_emb_dim=100,
        char_filters=0, use_charcnn=False, word_drop_alpha=DEFAULT_ALPHA))
    for wd, cd in [(30, 69), (50, 51), (70, 30), (85, 15)]:
        fpw = max(1, cd // 3)
        configs.append(AblationConfig(name=f"mix ({wd}+{cd})", word_emb_dim=wd,
            char_filters=fpw, char_widths=(2,3,4), use_charcnn=True,
            char_emb_dim=CHAR_EMB_DIM, word_drop_alpha=DEFAULT_ALPHA))
    configs.append(AblationConfig(name="no_word_dropout (70+30)", word_emb_dim=70,
        char_filters=10, char_widths=(2,3,4), use_charcnn=True, char_emb_dim=CHAR_EMB_DIM,
        word_drop_alpha=0.0))
    configs.append(AblationConfig(name="word_dropout_a=0.25 (70+30)", word_emb_dim=70,
        char_filters=10, char_widths=(2,3,4), use_charcnn=True, char_emb_dim=CHAR_EMB_DIM,
        word_drop_alpha=0.25))
    configs.append(AblationConfig(name="word_dropout_a=0.50 (70+30)", word_emb_dim=70,
        char_filters=10, char_widths=(2,3,4), use_charcnn=True, char_emb_dim=CHAR_EMB_DIM,
        word_drop_alpha=0.50))
    for h in [64, 128, 256, 512]:
        configs.append(AblationConfig(name=f"lstm_hidden={h} (70+30)", word_emb_dim=70,
            char_filters=10, char_widths=(2,3,4), use_charcnn=True, char_emb_dim=CHAR_EMB_DIM,
            hidden=h, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="BiLSTM word-only (100-dim, no CharCNN)",
        word_emb_dim=100, char_filters=0, use_charcnn=False, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="BiLSTM+CharCNN, no word dropout",
        word_emb_dim=64, char_emb_dim=32, char_filters=12, char_widths=(2,3,4),
        use_charcnn=True, word_drop_alpha=0.0))
    configs.append(AblationConfig(name="BiLSTM+CharCNN, word=76 char=24 (3x8 filters)",
        word_emb_dim=76, char_emb_dim=32, char_filters=8, char_widths=(2,3,4),
        use_charcnn=True, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="BiLSTM+CharCNN, word=88 char=12 (3x4 filters)",
        word_emb_dim=88, char_emb_dim=16, char_filters=4, char_widths=(2,3,4),
        use_charcnn=True, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="BiLSTM+CharCNN, 1-layer",
        word_emb_dim=64, char_emb_dim=32, char_filters=12, char_widths=(2,3,4),
        use_charcnn=True, n_layers=1, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="BiLSTM+CharCNN, hidden=128",
        word_emb_dim=64, char_emb_dim=32, char_filters=12, char_widths=(2,3,4),
        use_charcnn=True, hidden=128, word_drop_alpha=0.25))
    configs.append(AblationConfig(name="UniLSTM+CharCNN (unidirectional)",
        word_emb_dim=64, char_emb_dim=32, char_filters=12, char_widths=(2,3,4),
        use_charcnn=True, bidirectional=False, word_drop_alpha=0.25))
    return configs


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for word_ids, char_ids, tag_ids, lengths in loader:
        word_ids = word_ids.to(device)
        tag_ids  = tag_ids.to(device)
        if char_ids is not None:
            char_ids = char_ids.to(device)
        logits = model(word_ids, char_ids, lengths)
        B, T, K = logits.shape
        loss = criterion(logits.reshape(B * T, K), tag_ids.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()
        pred  = logits.argmax(-1)
        mask  = tag_ids != IGNORE_IDX
        correct += (pred[mask] == tag_ids[mask]).sum().item()
        total   += mask.sum().item()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for word_ids, char_ids, tag_ids, lengths in loader:
        word_ids = word_ids.to(device)
        tag_ids  = tag_ids.to(device)
        if char_ids is not None:
            char_ids = char_ids.to(device)
        logits = model(word_ids, char_ids, lengths)
        B, T, K = logits.shape
        loss = criterion(logits.reshape(B * T, K), tag_ids.reshape(B * T))
        total_loss += loss.item()
        pred  = logits.argmax(-1)
        mask  = tag_ids != IGNORE_IDX
        correct += (pred[mask] == tag_ids[mask]).sum().item()
        total   += mask.sum().item()
    return total_loss / len(loader), correct / total


def evaluate_on_test(model, test_sents, word2idx, char2idx, tag2idx, idx2tag,
                     device, use_charcnn=True, char_widths=(2, 3, 4)):
    model.eval()
    correct = total = 0
    iv_correct = iv_total = oov_correct = oov_total = 0
    sent_correct = 0
    vocab = set(word2idx.keys()) - {PAD_TOKEN, UNK_TOKEN}
    unk_word = word2idx[UNK_TOKEN]
    unk_char = char2idx.get("<cUNK>", 1)

    with torch.no_grad():
        for words, gold_tags in test_sents:
            word_ids = [word2idx.get(w, unk_word) for w in words]
            T = len(words)
            if use_charcnn:
                char_ids_list = [word_to_char_ids(w, char2idx) for w in words]
                L = max(len(c) for c in char_ids_list)
                L = max(L, max(char_widths))
                char_tensor = torch.zeros(1, T, L, dtype=torch.long, device=device)
                for t, ch in enumerate(char_ids_list):
                    char_tensor[0, t, :len(ch)] = torch.tensor(ch, dtype=torch.long)
            else:
                char_tensor = None
            ids_t  = torch.tensor([word_ids], dtype=torch.long, device=device)
            lens_t = torch.tensor([T], dtype=torch.long)
            logits    = model(ids_t, char_tensor, lens_t)
            pred_ids  = logits.argmax(-1)[0]
            pred_tags = [idx2tag[i.item()] for i in pred_ids]

            all_ok = True
            for w, pt, gt in zip(words, pred_tags, gold_tags):
                total += 1
                if pt == gt:
                    correct += 1
                else:
                    all_ok = False
                if w in vocab:
                    iv_total += 1
                    if pt == gt:
                        iv_correct += 1
                else:
                    oov_total += 1
                    if pt == gt:
                        oov_correct += 1
            if all_ok:
                sent_correct += 1

    acc     = correct / total * 100 if total else 0
    iv_acc  = iv_correct / iv_total * 100 if iv_total else 0
    oov_acc = oov_correct / oov_total * 100 if oov_total else 0
    sent_acc = sent_correct / max(len(test_sents), 1) * 100
    return acc, iv_acc, oov_acc, sent_acc


def run_experiment(name, train_sents, val_sents, test_sents,
                   word2idx, char2idx, tag2idx, idx2tag, word_freq,
                   device, max_epochs=15, **kw):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")

    vocab_size = len(word2idx)
    n_chars    = len(char2idx)
    n_tags     = len(tag2idx)

    use_charcnn = kw.get("use_charcnn", True)
    char_widths = kw.get("char_widths", (2, 3, 4))
    cw = tuple(char_widths) if use_charcnn else (2, 3, 4)

    train_ds = POSDataset(train_sents, word2idx, char2idx, tag2idx,
                          word_freq=word_freq, train_mode=True,
                          word_drop_alpha=kw.get("word_drop_alpha", DEFAULT_ALPHA),
                          use_charcnn=use_charcnn)
    val_ds   = POSDataset(val_sents,   word2idx, char2idx, tag2idx,
                          train_mode=False, use_charcnn=use_charcnn)

    collate  = make_collate(cw, use_charcnn=use_charcnn)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = AblationModel(
        vocab_size=vocab_size, n_chars=n_chars, n_tags=n_tags,
        word_emb_dim=kw.get("word_emb_dim", 64),
        char_emb_dim=kw.get("char_emb_dim", CHAR_EMB_DIM),
        char_filters=kw.get("char_filters", 12),
        char_widths=kw.get("char_widths", (2, 3, 4)),
        hidden=kw.get("hidden", 256),
        n_layers=kw.get("n_layers", 2),
        dropout=kw.get("dropout", 0.5),
        bidirectional=kw.get("bidirectional", True),
        use_charcnn=use_charcnn,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_state   = None
    train_losses, val_accs = [], []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_acc)
        train_losses.append(tr_loss)
        val_accs.append(vl_acc * 100)
        print(f"  E{epoch:02d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% "
              f"val_acc={vl_acc*100:.2f}%", flush=True)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - start
    print(f"  Done in {elapsed/60:.1f} min. Best val: {best_val_acc*100:.2f}%")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_acc, iv_acc, oov_acc, sent_acc = evaluate_on_test(
        model, test_sents, word2idx, char2idx, tag2idx, idx2tag,
        device, use_charcnn=use_charcnn, char_widths=cw)

    result = {
        "name": name,
        "test_acc": test_acc,
        "iv_acc": iv_acc,
        "oov_acc": oov_acc,
        "sent_acc": sent_acc,
        "best_val_acc": best_val_acc * 100,
        "total_params": total_params,
        "train_time_min": elapsed / 60,
        "train_losses": train_losses,
        "val_accs": val_accs,
    }
    print(f"  TEST ACC: {test_acc:.2f}%  IV: {iv_acc:.2f}%  OOV: {oov_acc:.2f}%  SENT: {sent_acc:.2f}%")
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python abalations.py data/train.tags data/dev.sents data/dev.tags [--epochs N] [--fast]")
        sys.exit(1)
    train_file = sys.argv[1]
    dev_sents_file = sys.argv[2] if len(sys.argv) > 2 else "data/dev.sents"
    dev_tags_file  = sys.argv[3] if len(sys.argv) > 3 else "data/dev.tags"
    epochs = 15
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        epochs = int(sys.argv[idx+1])
    if "--fast" in sys.argv:
        epochs = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, epochs={epochs}", flush=True)

    print("Loading data ...", flush=True)
    all_sents = parse_tagged_file(train_file)
    random.seed(SEED)
    random.shuffle(all_sents)
    split = int(0.90 * len(all_sents))
    train_sents = all_sents[:split]
    val_sents   = all_sents[split:]
    dev_tagged = parse_tagged_file(dev_tags_file) if os.path.exists(dev_tags_file) else []
    dev_sents_eval = [ws for ws, _ in dev_tagged]
    dev_tags_eval  = [ts for _, ts in dev_tagged]
    test_pairs = list(zip(dev_sents_eval, dev_tags_eval))

    word2idx, char2idx, tag2idx, idx2tag, word_freq = build_vocab(train_sents)
    print(f"  Vocab: {len(word2idx)} words, {len(char2idx)} chars, {len(tag2idx)} tags")

    configs = build_configs()
    results = []

    cfg_bar = tqdm(enumerate(configs, 1), total=len(configs), desc="Ablations", unit="cfg")
    for i, cfg in cfg_bar:
        print(f"\n[{i}/{len(configs)}] {cfg.name}")
        kw = dict(
            word_emb_dim=cfg.word_emb_dim,
            char_emb_dim=cfg.char_emb_dim,
            char_filters=cfg.char_filters,
            char_widths=cfg.char_widths,
            use_charcnn=cfg.use_charcnn,
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            word_drop_alpha=cfg.word_drop_alpha,
        )
        res = run_experiment(
            cfg.name, train_sents, val_sents, test_pairs,
            word2idx, char2idx, tag2idx, idx2tag, word_freq, device,
            max_epochs=epochs, **kw)
        results.append(res)

    print("\n\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    print(f"{'Configuration':<52} {'DevAcc%':>8} {'IV%':>7} {'OOV%':>7} {'Sent%':>7} {'Params':>10}")
    print("-"*80)
    for r in results:
        print(f"  {r['name']:<50} {r['test_acc']:>7.2f}% {r['iv_acc']:>6.2f}% "
              f"{r['oov_acc']:>6.2f}% {r['sent_acc']:>6.2f}% {r['total_params']:>10,}")

    os.makedirs("data", exist_ok=True)
    out_path = "data/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
