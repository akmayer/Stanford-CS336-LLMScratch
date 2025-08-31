import random
import numpy as np
import torch
import einx

def dataLoader(numpyIdxs, batch_size, context_length, device='cpu'):
    minIdx = 0
    maxIdx = numpyIdxs.shape[0] - context_length
    sampleStartIndices = np.random.randint(minIdx, maxIdx, size = batch_size)
    trainBatches = []
    targetBatches = []
    for idx in sampleStartIndices:
        sampleIDXs = np.arange(idx, idx + context_length + 1)
        sample = einx.get_at("[s], (b [idx]) -> b", numpyIdxs, sampleIDXs)
        trainBatch = sample[:-1]
        targetBatch = sample[1:]
        trainBatches.append(trainBatch)
        targetBatches.append(targetBatch)
    
    return torch.tensor(np.stack(trainBatches)).long().to(device), torch.tensor(np.stack(targetBatches)).long().to(device)