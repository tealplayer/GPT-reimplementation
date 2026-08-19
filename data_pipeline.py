import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')

# hyperparameters
batch_size = 64
block_size = 256

d_model = 384
num_heads = 6
n_layers = 6
d_ff = 4 * d_model

lr = 3e-4
max_iters = 5000
eval_interval = 500

class Head(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        Q = self.W_Q(x) # Query vector - x is the embedding vector
        K = self.W_K(x) # Key vector - sequence of vectors that can potentially answer the queries
        V = self.W_V(x) # Value vector(down-projection) - used to update the embeddings(EX: you want the embedding of one word that's related to another to cause a change to
                        # that other word that more specifically encodes the now better known description now that you have that new context)
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / (d_k**0.5) # computes a dot product to see how well each key matches each query(larger dot product = more aligned)
        
        # mask the scores()
        T = scores.shape[-1]
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)

        out = weights @ V # think of V as changing the vector of the previous embedding now that we have more context on that embedding
                          # for each column in the grid, you mulyiply each of the value vectors by the corresponding weight in that column(from the softmax)
                          # then we add together all of these rescaled values(the weight*value vector) in the column(i.e. each token) to update the embedding
                          # WARNING: the process described above is transposed so it's actually across rows rather than along columns
                          # EX: 'fluffy' and 'blue' adding more context to the next word 'creature' bc they have large proportions of the value vectors, while the other words get
                          # zeroed out
        return out # result vector

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        d_k = d_model // num_heads

        self.heads = nn.ModuleList([Head(d_model, d_k, d_k) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        head_outs = [h(x) for h in self.heads] # heads_out[h] is a single head's result vector [B, T, 64]

        out = torch.cat(head_outs, dim=-1) # we concatenate rather than add since all of these head's dimensions mean different things to each head, even tho they're all 64
        out = self.proj(out) # does an up-projection on all of these head's respective result vectors so that they live in the same space
        
        return out

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_network = nn.Sequential( # MLP layer
            nn.Linear(d_model, d_ff), # expand: d_model -> d_ff(4 * d_model)
            nn.ReLU(),
            nn.Linear(d_ff, d_model), # project back: d_ff -> d_model
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm(make sure to explain why I chose this instead of Post-norm) + residual connections
        x = x + self.attention(self.ln1(x)) 
        x = x + self.feed_forward_network(self.ln2(x))     
        
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, num_heads, n_layers, d_ff):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        model.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1) # chose sampling over argmax
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

sorted_chars = sorted(set(file_content))
vocab_size = len(sorted_chars)

ints = list(range(vocab_size))

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
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(split, batch_size, block_size)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()

    return out

model = GPT(vocab_size, d_model, block_size, num_heads, n_layers, d_ff)
model = model.to(device)

optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f'iterations: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')
    
    xb, yb = get_batch('train', batch_size, block_size)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)

print(decoder(out[0].tolist()))
