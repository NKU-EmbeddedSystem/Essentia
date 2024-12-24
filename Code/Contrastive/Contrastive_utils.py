import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 生成不包括正样本中数据的负样本 -> [batch_size, embedding_dim]
def generate_negative_index(index_in_pos, neg_size, upbondary=29999):
    negative_num = neg_size // 2
    index_in_pos = set(index_in_pos.tolist())
    # index_in_neg = random_numbers_generator(0, 60000, index_in_pos, negative_num)
    index_in_neg = torch.tensor(random_num(0, upbondary, index_in_pos, negative_num))
    return index_in_neg


# 生成剔除某些特定值的随机数
def random_num(start, end, excluded, batch_size):
    while True:
        batch = set(random.sample(range(start, end + 1), batch_size))
        filtered_batch = list(batch - (batch & excluded))
        while len(filtered_batch) < batch_size:
            num = random.randint(start, end)
            if num not in excluded:
                filtered_batch.append(num)
        return filtered_batch


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.critation = nn.CrossEntropyLoss()

    def forward(self, queries, positives, negatives):
        batch_size = queries.shape[0]
        neg_num = negatives.shape[0]

        # Normalize inputs
        queries = F.normalize(queries, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)

        # for dim >= 3
        queries = queries.view(batch_size, -1)
        positives = positives.view(batch_size, -1)
        negatives = negatives.view(neg_num, -1)

        # Calculate similarity scores
        pos_scores = torch.sum(queries * positives, dim=1) / self.temperature
        neg_scores = torch.mm(queries, negatives.t()) / self.temperature

        # Calculate the InfoNCE loss
        logits = torch.cat((pos_scores.unsqueeze(1), neg_scores), dim=1)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = self.critation(logits, targets)

        return loss


if __name__ == '__main__':
    pass
