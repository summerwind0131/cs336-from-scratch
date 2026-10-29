import json
from collections.abc import Iterable, Iterator
from os import PathLike

import regex

from cs336_basics.pretokenize import (
    PAT,
    pretokenize,
    split_special_tokens,
)
from cs336_basics.bpe_train import merge_word


def gpt2_byte_decoder() -> dict[str, int]:
    """
    GPT-2 byte-to-unicode 映射的逆映射。
    用于读取 GPT-2 格式的 vocab.json 和 merges.txt。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs.copy()
    existing = set(bs)
    n = 0

    for b in range(256):
        if b not in existing:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {
        chr(c): b
        for b, c in zip(bs, cs)
    }


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        self.byte_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        # 添加词表中尚不存在的特殊 token
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id

        self.special_token_set = set(self.special_tokens)

        # Merge 顺序决定优先级
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        加载 GPT-2 格式的 vocab.json 和 merges.txt。
        """
        decoder = gpt2_byte_decoder()

        def to_bytes(token: str) -> bytes:
            return bytes(decoder[c] for c in token)

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {
            token_id: to_bytes(token)
            for token, token_id in raw_vocab.items()
        }

        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line == "#version: 0.2":
                    continue

                parts = line.split()

                if len(parts) != 2:
                    continue

                a, b = parts
                merges.append((to_bytes(a), to_bytes(b)))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """
        将一个 pretoken 转成 byte-level BPE token IDs。
        使用 merge rank 决定每次合并的 pair。
        """
        word = tuple(
            bytes([b])
            for b in pretoken.encode("utf-8")
        )

        while len(word) > 1:
            best_pair = None
            best_rank = float("inf")

            for a, b in zip(word, word[1:]):
                pair = (a, b)
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            word = merge_word(word, best_pair)

        return [
            self.byte_to_id[token]
            for token in word
        ]

    def encode(self, text: str) -> list[int]:
        result = []

        chunks = split_special_tokens(
            text,
            self.special_tokens,
        )

        for chunk in chunks:
            if chunk in self.special_token_set:
                result.append(
                    self.byte_to_id[chunk.encode("utf-8")]
                )
                continue

            for pretoken in pretokenize(chunk):
                result.extend(
                    self._encode_pretoken(pretoken)
                )

        return result

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        流式编码，不一次性读取全部文本。

        保留尚未确定的文本后缀，以避免：
        1. 普通 pretoken 跨 chunk 边界
        2. 特殊 token 跨 chunk 边界
        3. 尾部空白的正则匹配因后续文本改变
        """
        buffer = ""

        max_special_len = max(
            (len(s) for s in self.special_tokens),
            default=1,
        )

        for chunk in iterable:
            buffer += chunk

            # 保留足够长的后缀，以检测跨块特殊 token
            safe_limit = max(
                0,
                len(buffer) - max_special_len + 1,
            )

            tokens = []
            offset = 0

            for part in split_special_tokens(
                buffer, self.special_tokens
            ):
                if part in self.special_token_set:
                    tokens.append(
                        (part, True, offset + len(part))
                    )
                else:
                    for match in regex.finditer(PAT, part):
                        tokens.append(
                            (
                                match.group(),
                                False,
                                offset + match.end(),
                            )
                        )

                offset += len(part)

            consumed = 0

            # 最后一个 token 暂时保留，避免跨块合并错误
            for token, is_special, end in tokens[:-1]:
                if end > safe_limit:
                    break

                if is_special:
                    yield self.byte_to_id[
                        token.encode("utf-8")
                    ]
                else:
                    yield from self._encode_pretoken(token)

                consumed = end

            buffer = buffer[consumed:]

        # 处理最后剩余的文本
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        encoded = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return encoded.decode(
            "utf-8",
            errors="replace",
        )