import torch
import torch.nn as nn
import torch.nn.functional as F

with open('input.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

class Head(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / (d_k**0.5)
        weights = F.softmax(scores, dim=-1)

        out = weights @ V
        return out

sorted_chars = sorted(set(file_content))

ints = list(range(len(sorted_chars)))

stoi = dict(zip(sorted_chars, ints)) # maps character : integer
itos = dict(zip(ints, sorted_chars)) # maps integer : character

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encoder(file_content), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data

    ii = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ii])
    y = torch.stack([data[i+1 : i + block_size + 1] for i in ii])
    return x, y

xb, yb = get_batch('train', 32, 256)

print(repr(decoder(xb[0].tolist())))
print(repr(decoder(yb[0].tolist())))

