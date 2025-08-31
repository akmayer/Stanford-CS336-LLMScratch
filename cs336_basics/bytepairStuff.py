import regex as re
from collections import defaultdict
from typing import Iterator, Iterable
from functools import cache, lru_cache
from tqdm import tqdm


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
    bytePairCountsSorted = sorted(bytePairCounts.items(), key = lambda x : (x[1], x[0]), reverse=True)
    
    merge = bytePairCountsSorted[0][0]
    pairCount = bytePairCountsSorted[0][1]
    bytePairCounts[merge] = 0
    return bytePairCounts, merge, pairCount


def mergePretokenizedCounts(pretokenizedCounts, mergedPair, bytePairCounts):
    newPretokenizedCounts = defaultdict(int)
    pretokensContainingMerge = []
    for pretoken, count in pretokenizedCounts.items():
        modifiedPretoken = list(pretoken)
        pretokenContainsMerge = False

        while True:
            mergeLocationTuple = pretokenContainsBytePair(modifiedPretoken, mergedPair)
            if not mergeLocationTuple[0]:
                break
            pretokenContainsMerge = True
            mergeLocation = mergeLocationTuple[1]
            if mergeLocation != 0:
                leftOverlap = (modifiedPretoken[mergeLocation - 1], modifiedPretoken[mergeLocation])
                if leftOverlap[0] != mergedPair[0] + mergedPair[1]:
                    bytePairCounts[leftOverlap] -= count
                    
            if mergeLocation + 1 != len(modifiedPretoken) - 1:
                rightOverlap = (modifiedPretoken[mergeLocation + 1], modifiedPretoken[mergeLocation + 2])
                bytePairCounts[rightOverlap] -= count
            
            modifiedPretoken[mergeLocation + 1] = mergedPair[0] + mergedPair[1]
            modifiedPretoken = modifiedPretoken[:mergeLocation] + modifiedPretoken[mergeLocation + 1:]

        
        
        modifiedPretoken = tuple(modifiedPretoken)
        newPretokenizedCounts[modifiedPretoken] = count
        if pretokenContainsMerge:
            for idx, byte in enumerate(modifiedPretoken[:-1]):
                if byte == (mergedPair[0] + mergedPair[1]) or modifiedPretoken[idx + 1] == (mergedPair[0] + mergedPair[1]):
                    bytePairCounts[(byte, modifiedPretoken[idx + 1])] += count
    return pretokensContainingMerge, newPretokenizedCounts

def getMerges(path, vocabSize, specialTokens):
    with open(path, 'r') as f:
        data = f.read()
        
    specialTokenPAT = '|'.join(re.escape(s) for s in specialTokens)
    sep_re = re.compile(specialTokenPAT)
    tok_re = re.compile(PAT)
    
    def iter_docs(text):
        start = 0
        for m in sep_re.finditer(text):
            yield text[start:m.start()]
            start = m.end()
        yield text[start:]
    
    # Count docs quickly without building them
    num_docs = sum(1 for _ in sep_re.finditer(data)) + 1
    
    pretokenizedCounts = defaultdict(int)
    print("Tokenizing Docs...")
    for doc in tqdm(iter_docs(data), total=num_docs):
        for x in tok_re.finditer(doc):
            bytesRep = tuple(k.to_bytes() for k in x[0].encode('utf-8'))
            pretokenizedCounts[bytesRep] += 1
    
        
    vocabulary = [x.to_bytes() for x in range(256)] + [x.encode('utf-8') for x in specialTokens]
    
    merges = []
    bytePairCounts, mergedPair, pairCount = makeMerge(pretokenizedCounts)
    print(mergedPair)
    
    num_merges = vocabSize - len(vocabulary)

    print("Merging...")
    for _ in tqdm(range(num_merges), total=num_merges, desc="Merging", disable=False):
        merges.append(mergedPair)
        vocabulary.append(mergedPair[0] + mergedPair[1])
    
        pretokensContainingMerge, pretokenizedCounts = mergePretokenizedCounts(
            pretokenizedCounts, mergedPair, bytePairCounts
        )
    
        bytePairCountsSorted = sorted(
            bytePairCounts.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True
        )
    
        mergedPair, pairCount = bytePairCountsSorted[0]
        bytePairCounts[mergedPair] = 0
    vocabDict = {}
    for idx, word in enumerate(vocabulary):
        vocabDict[idx] = word
    return vocabDict, merges

def pretokenContainsBytePair(pretokenTuple, bytePairTuple):
    for idx, byte in enumerate(pretokenTuple[:-1]):
        if byte == bytePairTuple[0] and pretokenTuple[idx + 1] == bytePairTuple[1]:
            return True, idx
    return False, 0

def tryMerge(pretoken, mergedPair):
    modifiedPretoken = list(pretoken)
    pretokenContainsMerge = False
    while True:
        mergeLocationTuple = pretokenContainsBytePair(modifiedPretoken, mergedPair)
        if not mergeLocationTuple[0]:
            break
        pretokenContainsMerge = True
        mergeLocation = mergeLocationTuple[1]
        
        modifiedPretoken[mergeLocation + 1] = mergedPair[0] + mergedPair[1]
        modifiedPretoken = modifiedPretoken[:mergeLocation] + modifiedPretoken[mergeLocation + 1:]
    modifiedPretoken = tuple(modifiedPretoken)
    return modifiedPretoken

@lru_cache(maxsize=20000)
def tokenizeWord(word, merges):
    modifiedPretoken = word
    for merge in merges:
        modifiedPretoken = tryMerge(modifiedPretoken, merge)
    return modifiedPretoken
    
class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = tuple(merges)
        self.special_tokens = special_tokens
        self.vocabToIdx = {v : k for k, v in self.vocab.items()}
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

            specialTokenPAT = "(" + "|".join(re.escape(s) for s in tokens_sorted) + ")"
            docs = re.split(specialTokenPAT, text)
        else:
            docs = [text]
        pretokenizedWords = []
        
        for doc in docs:
            if self.special_tokens is not None and doc in self.special_tokens:
                pretokenizedWords.append((doc.encode('utf-8'),))
            else:
                splitText = re.finditer(PAT, doc)
                for idx, x in enumerate(splitText):
                    bytesRep = tuple(k.to_bytes() for k in x[0].encode('utf-8'))
                    pretokenizedWords.append(bytesRep)
        idSequence = []
        for pretoken in pretokenizedWords:
            if self.special_tokens is not None and pretoken in self.special_tokens:
                idSequence.append(self.vocabToIdx[pretoken])
                continue
            modifiedPretoken = tokenizeWord(pretoken, self.merges)
            for tok in modifiedPretoken:
                idSequence.append(self.vocabToIdx[tok])
            
        
        return idSequence
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            for tok in self.encode(string):
                yield tok
    def decode(self, ids: list[int]) -> str:
        txt = b''
        for tokId in ids:
            byte = self.vocab[tokId]
            txt += byte
        return txt.decode('utf-8', errors='replace')