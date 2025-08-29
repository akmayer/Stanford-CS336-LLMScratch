import regex as re
from collections import defaultdict

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
    docs = re.split(specialTokenPAT, data)

    pretokenizedCounts = defaultdict(int)
    for doc in docs:
        splitText = re.finditer(PAT, doc)
        for x in splitText:
            bytesRep = tuple(k.to_bytes() for k in x[0].encode('utf-8'))
            pretokenizedCounts[bytesRep] += 1
    
        
    vocabulary = [x.to_bytes() for x in range(256)] + [x.encode('utf-8') for x in specialTokens]
    
    merges = []
    bytePairCounts, mergedPair, pairCount = makeMerge(pretokenizedCounts)
    while len(vocabulary) < vocabSize:
        
        merges.append(mergedPair)
        vocabulary.append(mergedPair[0] + mergedPair[1])
        pretokensContainingMerge, pretokenizedCounts = mergePretokenizedCounts(pretokenizedCounts, mergedPair, bytePairCounts)
        bytePairCountsSorted = sorted(bytePairCounts.items(), key = lambda x : (x[1], x[0]), reverse=True)
    
        mergedPair = bytePairCountsSorted[0][0]
        pairCount = bytePairCountsSorted[0][1]
        bytePairCounts[mergedPair] = 0
    vocabDict = {}
    for idx, word in enumerate(vocabulary):
        vocabDict[idx] = word
    return vocabDict, merges