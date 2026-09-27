import torch


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: CrossEntropyLoss between corcorrectsponding logits and tokens
        """
        # НЕ УВИДЕЛ НАСЛЕДОВАНИЕ И НАПИСАЛ САМ
        
        # device = outputs.device
        # probs = torch.softmax(outputs[:, :-1], dim=-1) #b t-1 v
        # token_probs = probs.gather(
        #     dim=-1,
        #     index=tokens[:, 1:].unsqueeze(-1)
        # ).squeeze(-1) #b t-1

        # sz = token_probs.size(1) #t-1
        # # повторяем B раз длины последовательностей и смотрим только там,
        # # где меньше истиной длины(-1 у tokens_lens нужно, чтобы шло правильное сравнение,
        # # тк 0 позиция предсказывает 1, а предпоследний предскажет последний)
        # all_lens = tokens_lens.to(device) - 1 
        # mask = torch.arange(sz, device=device).expand(len(token_probs), sz) < (all_lens).view(-1, 1) #b t-1
        
        # if loss_mask is not None:
        #     mask *= loss_mask
        
        # cross_entropy = -torch.log(token_probs) * mask #b t-1
        # return cross_entropy.sum() / mask.sum()
        
        
        #УВИДЕЛ НАСЛЕДОВАНИЕ
        b, t, v = outputs.shape
        outputs = outputs[:, :-1]
        targets = tokens[:, 1:]
        device = outputs.device
 
        
        all_lens = tokens_lens.to(device) - 1 
        sz = t - 1
        mask = torch.arange(sz, device=device).expand(b, sz) < (all_lens).view(-1, 1) #b t-1
        mask = mask.to(bool)
        if loss_mask is not None:
            mask &= loss_mask
            
        outputs = outputs[mask]
        targets = targets[mask]
        return super().forward(outputs, targets)
        
    

class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: Accuracy for given logits and tokens
        """
        device = outputs.device
        ids = torch.argmax(outputs[:, :-1], dim=-1)
        corrects = (ids == tokens[:, 1:])
        sz = corrects.size(1)
        
        # повторяем B раз длины последовательностей и смотрим только там,
        # где меньше истиной длины(-1 у tokens_lens нужно, чтобы шло правильное сравнение,
        # тк 0 позиция предсказывает 1, а предпоследний предскажет последний)
        all_lens = tokens_lens.to(device) - 1
        mask = torch.arange(sz, device=device).expand(len(corrects), sz) < (all_lens).view(-1, 1)
        mask = mask.to(bool)

        if loss_mask is not None:
            mask &= loss_mask
            
        corr_masked = corrects[mask]
        
        return corr_masked.sum() / (mask).sum()
        
        
  
        
