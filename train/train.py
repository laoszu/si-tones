import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.aishell_dataset import AishellDataset
from data.collate import collate_fn
from utils.tokenizer import Tokenizer
from models.confromer.conformer import Conformer # XDD

import os
import json

if __name__ == '__main__':

    with open("config.json") as f:
        cfg = json.load(f)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer()
    tokenizer.load_vocab(cfg["vocab_path"])

    # and the data
    aishell_ds = AishellDataset(cfg["data_path"], sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"])

    aishell_dl = DataLoader(
        dataset=aishell_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = Conformer(
        input_dim=cfg["n_mels"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        conv_kernel=cfg["conv_kernel"],
        dropout=cfg["dropout"],
        vocab_size=len(tokenizer)
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
 
    d, w = cfg["d_model"], cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: d ** -0.5 * min(max(step, 1) ** -0.5, max(step, 1) * w ** -1.5),
    )
 
    ctc_loss = nn.CTCLoss(blank=tokenizer.char_to_id["<blank>"], zero_infinity=True)

    start_epoch = 0
    if os.path.exists(cfg["checkpoint"]):
        ckpt = torch.load(cfg["checkpoint"], map_location=dev)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"Wznowiono od epoki {start_epoch}")


    # training part
    for epoch in range(start_epoch + 1, cfg["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
 
        for i, (mels, mel_lengths, targets, target_lengths) in enumerate(aishell_dl):
            mels, mel_lengths = mels.to(dev), mel_lengths.to(dev)
            targets, target_lengths = targets.to(dev), target_lengths.to(dev)
 
            optimizer.zero_grad()
            log_probs, out_lengths = model(mels, mel_lengths)
            loss = ctc_loss(log_probs, targets, out_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
 
            total_loss += loss.item()
 
            if i % 50 == 0:
                print(f"epoch {epoch} | step {i}/{len(aishell_dl)} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
 
        print(f"epoch {epoch} | avg loss {total_loss / len(aishell_dl):.4f}")
 
        os.makedirs(os.path.dirname(cfg["checkpoint"]), exist_ok=True)
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, cfg["checkpoint"])

 

