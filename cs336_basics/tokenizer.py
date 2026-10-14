from collections import Counter

from cs336_basics.pretokenize import pretokenize,split_special_tokens
from cs336_basics.bpe_train import train_bpe,merge_word


class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.special_tokens=special_tokens or []
        self.merges=merges
        self.vocab=vocab

        self.byte_to_id={
            token:token_id for token_id,token in vocab.items()
        }
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        vocab_filepath:str

    def encode(self,text:str)->list[int]:

        result=[]
        """首先预处理 将text转换为pretoken"""
        chunks=split_special_tokens(text,self.special_tokens)
        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append(
                    self.byte_to_id[chunk.encode("utf-8")]
                )
                continue
            for pretoken in pretokenize(chunk):
                encoded=pretoken.encode("utf-8")
                word=tuple(bytes([b])for b in encoded)
                for merge in self.merges:
                    word=merge_word(word,merge)
                for token in word:
                    result.append(
                        self.byte_to_id[token]
                    )
                
        
        return result


    def encode_iterable(self,iterable):
        raise NotImplementedError

    def decode(self,ids:list[int])->str:
       encoded=b""
       for id in ids:
           encoded+=self.vocab[id]

       return encoded.decode("utf-8",errors="replace")
     