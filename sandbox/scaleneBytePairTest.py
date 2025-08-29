from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import regex as re
from collections import defaultdict


from scalene import scalene_profiler

# Turn profiling on


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenContainsBytePair(pretokenTuple, bytePairTuple):
    for idx, byte in enumerate(pretokenTuple[:-1]):
        if byte == bytePairTuple[0] and pretokenTuple[idx + 1] == bytePairTuple[1]:
            return True, idx
    return False, 0

def makeMerge(pretokenizedCounts):
    bytePairCounts = defaultdict(int)
    for pretoken, count in pretokenizedCounts.items():
        for idx, byte in enumerate(pretoken[:-1]):
            bytePairCounts[(byte, pretoken[idx + 1])] += count
    
    bytePairCounts = sorted(bytePairCounts.items(), key = lambda x : (x[1], x[0]), reverse=True)
    
    merge = bytePairCounts[0][0]
    return merge


def mergePretokenizedCounts(pretokenizedCounts, mergedPair):
    newPretokenizedCounts = defaultdict(int)
    for pretoken, count in pretokenizedCounts.items():
        modifiedPretoken = list(pretoken)
        
        while True:
            mergeLocationTuple = pretokenContainsBytePair(modifiedPretoken, mergedPair)
            if not mergeLocationTuple[0]:
                break
            mergeLocation = mergeLocationTuple[1]
            modifiedPretoken[mergeLocation + 1] = mergedPair[0] + mergedPair[1]
            modifiedPretoken = modifiedPretoken[:mergeLocation] + modifiedPretoken[mergeLocation + 1:]
        modifiedPretoken = tuple(modifiedPretoken)
        newPretokenizedCounts[modifiedPretoken] = count
    return newPretokenizedCounts

def getMerges(path, vocabSize, specialTokens):
    with open(path, 'r') as f:
        data = f.read()
        
    specialTokenPAT = '|'.join(re.escape(s) for s in specialTokens)
    docs = re.split(specialTokenPAT, data)

    pretokenizedCounts = defaultdict(int)
    for doc in docs:
        splitText = re.finditer(PAT, doc)
        for x in splitText:
            bytesRep = tuple(k.to_bytes() for k in x[0].encode('utf-8'))
            pretokenizedCounts[bytesRep] += 1
    
        
    vocabulary = [x.to_bytes() for x in range(256)] + [x.encode('utf-8') for x in specialTokens]
    
    merges = []
    while len(vocabulary) < vocabSize:
        mergedPair = makeMerge(pretokenizedCounts)
        merges.append(mergedPair)
        vocabulary.append(mergedPair[0] + mergedPair[1])
        pretokenizedCounts = mergePretokenizedCounts(pretokenizedCounts, mergedPair)

    vocabDict = {}
    for idx, word in enumerate(vocabulary):
        vocabDict[idx] = word
    return vocabDict, merges
scalene_profiler.start()

getMerges("../tests/fixtures/corpus.en", 500, ["<|endoftext|>"])


# Turn profiling off
scalene_profiler.stop()