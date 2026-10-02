import torch
import time
from torch import nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Callable

from argparse import ArgumentParser


from torch.utils.data import DataLoader
import data_utils
from positional_encoding import PositionalEncoding
from functools import partial
from transformers import AutoTokenizer

import amp
import utils


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class GPT2LikeModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1020, num_heads=4, max_length=data_utils.MAX_LENGTH):
        super().__init__()
        self.num_heads = num_heads
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_length)
        self.hidden_projector = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_projector2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.output_linear = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask):
        x = x.transpose(0, 1) # as we don't use batch first
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.hidden_projector(x)
        y = self.hidden_projector2(x)
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones((x.size(0), x.size(0)), dtype=torch.bool, device=x.device)
            )
        else:
            attention_mask = attention_mask.to(x.device)
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(
                attention_mask.logical_not(), float("-inf")
            )
        attention_mask = attention_mask.to(x.dtype)

        out = self.decoder(tgt=x, memory=x, tgt_mask=attention_mask, memory_mask=attention_mask)
        out = self.output_linear(out)
        return out.transpose(0, 1)


def get_gpt2_model(vocab_size) -> torch.nn.Module:
    return GPT2LikeModel(vocab_size)


def get_dataloader(dataloader_type, batch_size, path, k):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataloader = None
    if dataloader_type == 'base':
        start = time.perf_counter()
        ds = data_utils.BaseDataset(path, tokenizer)
        elapsed_time = time.perf_counter() - start
        collate = partial(data_utils.base_collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    if dataloader_type == 'standard':
        ### YOUR CODE HERE
        # prepare dataloader with StandardDataset and collate_fn
        start = time.perf_counter()
        ds = data_utils.StandardDataset(data_path=path, tokenizer=tokenizer)
        elapsed_time = time.perf_counter() - start
        collate = partial(data_utils.collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    if dataloader_type == 'balanced':
        ### YOUR CODE HERE
        # prepare dataloader with StandardDataset collate_fn and BalancedBatchSampler
        start = time.perf_counter()
        ds = data_utils.StandardDataset(data_path=path, tokenizer=tokenizer)
        sampler = data_utils.BalancedBatchSampler(dataset=ds, k=k, batch_size=batch_size)
        elapsed_time = time.perf_counter() - start
        collate = partial(data_utils.collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, collate_fn=collate, batch_sampler=sampler)
    if dataloader_type == 'sequenced':
        ### YOUR CODE HERE
        # prepare SequencedDataset
        start = time.perf_counter()
        ds = data_utils.SequencedDataset(data_path=path, tokenizer=tokenizer, batch_size=batch_size)
        elapsed_time = time.perf_counter() - start
        dataloader = DataLoader(ds, batch_size=None, collate_fn=ds.collate_data)
    return dataloader, elapsed_time


def train_epoch(
    train_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    metric: Callable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kind: str,
    scaler: None | amp.StaticGradScaler | amp.DynamicGradScaler,
    experiment_kind: str,
) -> None:
    model.train()
    pbar = tqdm(enumerate(train_loader))
    loss_accum = []
    acc_accum = []
    measurements_grad = []
    peak_gib = None
    num_steps = None


    for i, data in pbar:
        tokens, tokens_lens, attention_mask = data['tokens'].to(device), data['lengthes'], data['attention_mask']
        loss_mask = data.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        optimizer.zero_grad(set_to_none=False)
        # Obtain outputs and loss depending on kind. Plain fp16 should run without
        # Autocast; static/dynamic modes should use the custom Autocast.
        # Pass loss_mask to the criterion for sequenced batches.
        
        if kind == 'fp16' or kind == 'fp32':
            # compute grads without scaling
            if experiment_kind == 'amp':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs = model(tokens, attention_mask)
                loss = criterion(outputs, tokens, tokens_lens, loss_mask)
                loss.backward()
                end.record()
                torch.cuda.synchronize()
                measurements_grad.append(start.elapsed_time(end) / 1000)
                optimizer.step()
            else:
                outputs = model(tokens, attention_mask)
                loss = criterion(outputs, tokens, tokens_lens, loss_mask)
                loss.backward()
                optimizer.step()
        else:
            if experiment_kind == 'amp':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                autocast = amp.Autocast()
                with autocast:
                    start.record()
                    outputs = model(tokens, attention_mask)
                    loss = criterion(outputs, tokens, tokens_lens, loss_mask)
                # compute grads with scaling
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                end.record()
                torch.cuda.synchronize()
                measurements_grad.append(start.elapsed_time(end) / 1000)
                scaler.step(optimizer)
                scaler.update()
            else:
                autocast = amp.Autocast()
                with autocast:
                    outputs = model(tokens, attention_mask)
                    loss = criterion(outputs, tokens, tokens_lens, loss_mask)
                # compute grads with scaling
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()


        accuracy = metric(outputs, tokens, tokens_lens, loss_mask=loss_mask)
        num_steps = i
        loss_cpu = loss.item()
        accuracy_cpu = accuracy.item()
        if experiment_kind == 'amp':
            loss_accum.append(loss_cpu)
            acc_accum.append(accuracy_cpu)

        # первое замедление, поменять на detach
        pbar.set_description(f"Loss: {round(loss_cpu, 4)} " f"Accuracy: {round(accuracy_cpu * 100, 4)}")
    if experiment_kind == 'loading':
        return num_steps + 1
    else:
        return loss_accum, acc_accum, measurements_grad, num_steps + 1

def parse_args():
    parser = ArgumentParser(description="Training Acceleration Task")

    parser.add_argument("--exp", choices=['amp', 'loading'], default='amp', help='which type of experiment')
    parser.add_argument("--kind", choices=["fp32", "fp16", "static", "dynamic"], default="fp32", help="Training kind (dtype and scaler)")
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of epochs')
    parser.add_argument("--scale", type=float, default=65536.0, help='Initial/static loss scale')
    parser.add_argument("--factor", type=float, default=2.0, help='Dynamic loss scale multiplier')
    parser.add_argument("--patience", type=int, default=80, help='Patience for DynamicGradScaler')
    parser.add_argument("--min_scale", type=float, default=1.0, help='Minimum dynamic loss scale')
    parser.add_argument("--max_scale", type=float, default=2.0**24, help='Maximum dynamic loss scale')

    parser.add_argument("--dataloader", choices=["base", "standard", "balanced", "sequenced"], default="base", help="Dataloader type")
    parser.add_argument("--path", type=str, default='data/validation-00000-of-00001.txt', help='Dataset path')
    parser.add_argument("--k", type=int, default=20, help='k for balanced')

    return parser.parse_args()


def train():
    set_global_seed(42)
    args = parse_args()
    device = torch.device("cuda:0")
    dataloader, elapsed_time = get_dataloader(args.dataloader, args.batch_size, args.path, args.k)
    model = get_gpt2_model(dataloader.dataset.tokenizer.vocab_size).to(device)
    if args.kind == 'fp16':
        model = model.half()
    criterion = utils.LMCrossEntropyLoss()
    metric = utils.LMAccuracy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = None
    if args.kind == 'static':
        # Initialize StaticGradScaler with args.scale.
        scaler = amp.StaticGradScaler(args.scale)
    elif args.kind == 'dynamic':
        # Initialize DynamicGradScaler with the scale-related CLI arguments.
        scaler = amp.DynamicGradScaler(
            args.scale, 
            factor=args.factor, 
            patience=args.patience, 
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    num_epochs = args.num_epochs

    all_loss, all_acc, all_grad_time_measure = [], [], []
    all_steps = 0
    if args.exp == 'loading':
        torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(0, num_epochs):
        if args.exp == 'amp':
            loss, accuracy, grad_time, num_steps = train_epoch(
                train_loader=dataloader, 
                model=model, 
                criterion=criterion, 
                metric=metric,
                optimizer=optimizer, 
                device=device, 
                kind=args.kind, 
                scaler=scaler,
                experiment_kind=args.exp
            )

            all_loss.extend(loss)
            all_acc.extend(accuracy)
            all_grad_time_measure.extend(grad_time)
            all_steps += num_steps
        else:
            num_steps = train_epoch(
                            train_loader=dataloader, 
                            model=model, 
                            criterion=criterion, 
                            metric=metric,
                            optimizer=optimizer, 
                            device=device, 
                            kind=args.kind, 
                            scaler=scaler,
                            experiment_kind=args.exp
                        )
            all_steps += num_steps
    torch.cuda.synchronize()
    all_train_time = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    if args.exp == 'amp':
        with open(args.kind + '.json', 'w', encoding='utf-8') as f:
            # loss, accuracy, grad_time, all_steps
            results = {
                'loss_by_steps': all_loss,
                'accuracy_by_steps': all_acc,
                'grad_time_by_steps': all_grad_time_measure,
                'all_steps': all_steps
            }
            json.dump(results, f)
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

        axes[0].plot(range(1, len(all_loss) + 1), all_loss)
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{args.kind}: качество и время вычисления градиентов")

        axes[1].plot(range(1, len(all_acc) + 1), all_acc)
        axes[1].set_ylabel("Accuracy")

        grad_time_ms = [seconds * 1000 for seconds in all_grad_time_measure]
        axes[2].plot(range(1, len(grad_time_ms) + 1), grad_time_ms)
        axes[2].set_ylabel("Время, мс")

        axes[2].set_xlabel("Шаг обучения")
        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.savefig(f"{args.kind}.png", dpi=150)
        plt.close(fig)
        
    else:
        with open(args.dataloader + '.json', 'w', encoding='utf-8') as f:
            # elapsed_time, peak_gib, all_train_time, all_steps
            results = {
                'dataloader_time': elapsed_time,
                'peak_mem_allocation': peak_gib,
                'all_train_time': all_train_time,
                'all_steps': all_steps
            }
            json.dump(results, f)


if __name__ == '__main__':
    train()
