from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F


res_path = Path('res')
abs_res_path = res_path.resolve()

# TODO make sure CUDA is avialable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open(Path(abs_res_path, 'input.txt'), 'r', encoding='utf-8') as file:
    text = file.read()


chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
print(stoi)

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)


print(encode("Hi there"))
print(decode(encode("hii there")))
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


batch_size = 4
block_size = 8

def get_batch(split):
    data =  train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        
        logits = self.token_embedding_table(inputs) # (batch, block, vocab) other way of putting it (batch, time, channel)B,T,C

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    def generate(self, inputs, max_new_tokens):
        for i in range(max_new_tokens):
            logits, loss = self(inputs)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softamx to get distributions
            probs = F.softmax(logits, dim=1) # (B, C)
            output = torch.multinomial(probs, num_samples=1) # (B, 1)
            inputs = torch.cat((inputs, output), dim=1) # (B, T+1)
        return inputs
    
model = BigramLanguageModel(vocab_size)
# logits, loss = model(xb, yb)
# print(logits.shape)
# print(loss)

# inputs = torch.zeros((1, 1), dtype=torch.long) # it is 0 == new line character '/n'
# print(decode(model.generate(inputs, 
#                             max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 1000 == 0:
        print(f"Step {steps} loss: {loss.item()}")

print(loss.item())

inputs = torch.zeros((1, 1), dtype=torch.long) # it is 0 == new line character '/n'
print(decode(model.generate(inputs, 
                            max_new_tokens=500)[0].tolist()))

# TODO continue with the work, and build a proper transformer