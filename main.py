from pathlib import Path


res_path = Path('res')
abs_res_path = res_path.resolve()

with open(Path(abs_res_path, 'input.txt'), 'r', encoding='utf-8') as file:
    text = file.read()


chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)
print("")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)


print(encode("Hi there"))
print(decode(encode("hii there")))