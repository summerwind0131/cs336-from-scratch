from collections import Counter
from os import PathLike

from cs336_basics.pretokenize import pretokenize,split_special_tokens



Word=tuple[bytes,...]
Pair=tuple[bytes,bytes]

def count_pairs(
        words:dict[Word,int]
)->Counter[Pair]:
    pair_counts=Counter()
    for word,times in words.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            pair_counts[pair]+=times

    return pair_counts


def merge_word(
        word:Word,
        pair:Pair
)->Word:
    """将一个token序列指定的pair合并"""
    result=[]
    i=0

    while i <len(word):
        if(i+1<len(word) and (word[i],word[i+1])==pair):
            result.append(word[i]+word[i+1])
            i+=2
        else:
            result.append(word[i])
            i+=1

    return tuple(result)

def merge_words(
        words:dict[Word,int],
        pair:Pair
)->dict[Word,int]:
    new_words={}
    for word,times in words.items():
        if pair[0] in word and pair[1] in word:
            new_word=merge_word(word,pair)
            new_words[new_word]=new_words.get(new_word,0)+times
        else:
            new_words[word]=new_words.get(word,0)+times

    return new_words

def train_bpe(
        input_path:str|PathLike,
        vocab_size:int,
        special_tokens:list[str]
)->tuple[dict[int,bytes],list[Pair]]:

    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")

    merges:list[Pair]=[]
    with open(input_path,"r",encoding="utf-8") as f:
        text=f.read()

    words=Counter()

    chunks = split_special_tokens(text, special_tokens)
    special_token_set = set(special_tokens)
    for chunk in chunks:
        if chunk in special_token_set:
            continue
        for pretoken in pretokenize(chunk):
            encoded = pretoken.encode("utf-8")
            word = tuple(bytes([b]) for b in encoded)
            words[word] += 1

    while len(vocab)<vocab_size:
        pair_counts=count_pairs(words)
        if not pair_counts:
            break
        best_pair=max(
            pair_counts,
            key=lambda p:(pair_counts[p],p)
        )
        words=merge_words(words,best_pair)
        merges.append(best_pair)
        vocab[len(vocab)]=best_pair[0]+best_pair[1]

    return vocab,merges


if __name__ == "__main__":
    from pathlib import Path

    Path("toy.txt").write_text(
        "abababab",
        encoding="utf-8"
    )

    vocab, merges = train_bpe(
        input_path="toy.txt",
        vocab_size=257,
        special_tokens=[],
    )

    print("Vocab size:", len(vocab))
    print("Merges:", merges)
    print("New token:", vocab[256])

    assert len(vocab) == 257
    assert merges == [(b"a", b"b")]
    assert vocab[256] == b"ab"

    print("All tests passed!")