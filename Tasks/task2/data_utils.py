from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from collections import defaultdict
import random


MAX_LENGTH = 512


class BaseDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        el = self.samples[idx]
        input_ids = self.tokenizer(el)['input_ids']
        input_ids = input_ids[:self.max_length]
        length = len(input_ids)
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.int64), length



class StandardDataset(Dataset):
    """
    See task desciption
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        input_ids = self.tokenizer(sample)['input_ids']
        input_ids = input_ids[:self.max_length]
        length = len(input_ids)
        return torch.tensor(input_ids, dtype=torch.int64), length
        


def base_collate_fn(
    batch: list[tuple[torch.Tensor, int]],
    pad_token_id: int = 0
) -> dict:
    batch_processed = torch.stack([el[0] for el in batch])
    lengthes = torch.tensor([el[1] for el in batch])
    return {
        'tokens': batch_processed, 
        'lengthes': lengthes, 
        'attention_mask': None,
        'loss_mask': None,
    }


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
    pad_token_id: int = 0
) -> dict:
    """
    See task desciption
    """
    lengths = [el[1] for el in batch]
    samples = [el[0] for el in batch]
    max_len = max(lengths)
    for i in range(len(samples)):
        if len(samples[i]) < max_len:
            pad_tensor = torch.full((max_len - len(samples[i]), ), pad_token_id, 
                                    dtype=samples[i].dtype, device=samples[i].device)
            samples[i] = torch.cat([samples[i], pad_tensor], dim=-1)

    return {
        'tokens': torch.stack(samples),
        'lengthes': torch.tensor(lengths),
        'attention_mask': None,
        'loss_mask': None,
    }



# Для следующего подхода реализуйте BalancedBatchSampler, который объединяет в один батч примеры похожей длины. 
# Батчи должны быть случайными, а разница между самой длинной и самой короткой последовательностью в батче — не больше k. 
# Часть батчей может быть короче заданного размера. Удобно унаследоваться от Sampler. 
# Метод __init__ должен работать за линейное от размера датасета время, а получение следующего батча — за время, 
# линейное по размеру батча. Для этого можно хранить в хеш-таблице соответствие длины примера и его индексов.


class BalancedBatchSampler(Sampler):
    """
    See task desciption
    """
    def __init__(self, dataset, k: int, batch_size: int):
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size

        # артефакты прошлого решения
        # self.len_to_idx = {}

        # for idx, (sample, length) in enumerate(dataset):
        #     self.len_to_idx.setdefault(length, []).append(idx)

        self.buckets = {}
        for idx, (sample, length) in enumerate(dataset):
            self.buckets.setdefault((length // (k + 1)), []).append(idx)

        self.num_batches = sum((len(indices) + batch_size - 1) // batch_size for indices in self.buckets.values())

        
        

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        keys = list(self.buckets.keys())
        random.shuffle(keys)
        for key in keys:
            indices = self.buckets[key]
            n = len(indices)
            shuffled_idxs = random.sample(list(indices), k=n)

            for i in range(0, n, self.batch_size):
                idxes = shuffled_idxs[i : i + self.batch_size]
                yield idxes

        # рабочий код, но невозможно заранее узнать кол-во батчей, 
        # так как k фиксируется сразу от выбранной длины и не меняется динамически

        # shuffled_lengths = random.sample(list(self.len_to_idx.keys()), k=len(self.len_to_idx))
        # used = set()
        # for ln in shuffled_lengths:
        #     if ln in used: continue
        #     example_indices = []
        #     lens = list(range(ln, ln + self.k + 1))
        #     for i in lens:
        #         if i in self.len_to_idx:
        #             if i in used: continue
        #             random.shuffle(self.len_to_idx[i])
        #             for j in range(len(self.len_to_idx[i]) - 1, -1, -1):
        #                 example_indices.append(self.len_to_idx[i][j])
        #                 if len(example_indices) == self.batch_size:
        #                     yield example_indices
        #                     example_indices = []
        #             used.add(i)
        #     if example_indices:
        #         yield example_indices



# - `SequencedDataset` собирает примеры батча в одну длинную последовательность. 
# Он должен возвращать блочно-каузальную `attention_mask`, которая запрещает смотреть в будущее и между исходными примерами, 
# а также `loss_mask` формы `[1, T - 1]`, которая исключает из loss и accuracy переходы через границы примеров. 
# Если последний пример слишком длинный, его можно обрезать или отбросить. 
# Для этой реализации используйте `IterableDataset`.

class SequencedDataset(IterableDataset):
    """
    See task desciption
    """
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        batch_size: int, 
        max_length: int = MAX_LENGTH
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

        self.samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(line)


    
    def __iter__(self):
        result = []
        total_len = 0
        mask = []
        num_elems = 0

        for sample in self.samples:
            ids = self.tokenizer(sample)['input_ids']
            ln = len(ids)
            if total_len + ln >= self.max_length:
                result.extend(ids[:self.max_length - total_len])
                mask.append(self.max_length)
                yield torch.tensor(result).unsqueeze(0), mask, self.max_length

                result = []
                total_len = 0
                num_elems = 0
                mask = []
            elif num_elems + 1 == self.batch_size:
                end = min(ln, self.max_length - total_len)
                result.extend(ids[:end])
                total_len += end
                mask.append(total_len)
                yield torch.tensor(result).unsqueeze(0), mask, total_len

                result = []
                total_len = 0
                num_elems = 0
                mask = []
            else:
                total_len += ln
                num_elems += 1
                result.extend(ids)
                mask.append(total_len)

        if result:
            yield torch.tensor(result).unsqueeze(0), mask, total_len



    def collate_data(self, batch):
        # Return a loss_mask in addition to the block-causal attention mask.
        # It must exclude transitions between adjacent packed samples.
        ids, mask_intervals, total_len = batch

        loss_mask = torch.ones(total_len - 1, dtype=torch.bool)
        loss_intervals = torch.tensor(mask_intervals[:-1], dtype=torch.long) - 1
        loss_mask[loss_intervals] = False
        loss_mask = loss_mask.unsqueeze(0)

        attention_mask = torch.zeros((total_len, total_len), dtype=torch.bool)
        start = 0
        for idx in mask_intervals:
            matrix = torch.ones(idx - start, idx - start, dtype=torch.bool)
            lower_tri = torch.tril(matrix)
            attention_mask[start:idx, start:idx] = lower_tri
            start = idx

        return {
            'tokens': ids,
            'lengthes': torch.tensor(total_len),
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
        }
