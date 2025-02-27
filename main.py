import torch
from torch import nn

max_context_size = 256
n_embed = 384
n_heads = 6
n_layer = 6
dropout = 0.2
device = "cpu"

checkpoint = torch.load("bigram_lm_checkpoint.pth", map_location=device)
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_chars_size = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Create a lower triangular matrix for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(max_context_size, max_context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scale and compute dot-product attention weights
        weights = q @ k.transpose(-2, -1) * (C ** 0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_chars_size, n_embed)
        self.position_embedding = nn.Embedding(max_context_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_chars_size)

    def forward(self, x, y=None):
        B, T = x.shape
        token_embeddings = self.embedding(x)
        pos_embedding = self.position_embedding(torch.arange(T, device=x.device))
        x = token_embeddings + pos_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if y is None:
            loss = None
        else:
            # Flatten the logits and targets for cross-entropy loss calculation
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            y = y.view(B * T)
            loss = nn.functional.cross_entropy(logits, y)
        return logits, loss

    def generate(self, idx, new_tokens_count):
        # Generate new tokens iteratively
        for _ in range(new_tokens_count):
            idx_cond = idx[:, -max_context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            print(decode([next_token.item()]), end='', flush=True)


model = BigramLM().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


init_token = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = model.generate(init_token, new_tokens_count=800)[0].tolist()

print(decode(generated_sequence))

# for i in range(max_training_iterations):
#
#     if i % eval_interval == 0:
#         losses = estimate_loss()
#         print(f'step {i}: train loss = {losses["train"]:.4f}, eval loss = {losses["eval"]:.4f}')
#
#     x_batch, y_batch = get_batch('train')
#     _, loss = model(x_batch, y_batch)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()