import re
import regex

PAT=(
   r"'(?:[sdmt]|ll|ve|re)"
   r"| ?\p{L}+"
   r"| ?\p{N}+"
   r"| ?[^\s\p{L}\p{N}]+"
   r"|\s+(?!\S)"
   r"|\s+" 
)

def split_special_tokens(
        text:str,
        special_tokens:list[str]
)->list[str]:
    if not special_tokens:
        return [text]

    pattern="|".join(
        re.escape(token)
        for token in sorted(special_tokens,key=len,reverse=True)
    )
    return re.split(pattern,text)


def pretokenize(text:str)->list[str]:
    pretokens=[]
    for match in regex.finditer(PAT,text):
        pretokens.append(match.group())

    return pretokens


if __name__ =="__main__":
    text="Hello world! I love Python."
    tokens=pretokenize(text)
    for token in tokens:
        print(repr(token))

    assert tokens == [
        "Hello",
        " world",
        "!",
        " I",
        " love",
        " Python",
        ".",
    ]

    # 测试 2：特殊 token 切分
    text = "hello<|endoftext|>world"
    chunks = split_special_tokens(
        text,
        ["<|endoftext|>"]
    )

    print("Chunks:", chunks)
    assert chunks == ["hello", "world"]

    print("All tests passed!")