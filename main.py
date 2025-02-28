import torch
from torch import nn

torch.manual_seed(1337)
max_context_size = 256
batch_size = 64
learning_rate = 0.001
max_training_iterations = 10000
eval_interval = 100
generation_interval = 500
eval_iter = 200
n_embed = 384
n_heads = 4
n_layer = 4
dropout = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('train_data', 'r', encoding='utf8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_chars_size = len(chars)

# TODO: change encoder and decoder to tiktoken
encode = lambda e: [stoi[char] for char in e]
decode = lambda d: ''.join([itos[idx] for idx in d])

data = torch.tensor(encode(text))

data_split = int(len(text) * 0.9)
train_data = data[:data_split]
eval_data = data[data_split:]


def get_batch(mode):
    tr_data = train_data if mode == 'train' else eval_data
    ix = torch.randint(len(tr_data) - max_context_size, (batch_size,))
    x = torch.stack([tr_data[i:i + max_context_size] for i in ix])
    y = torch.stack([tr_data[i + 1:i + max_context_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    res = {}
    model.eval()
    for mode in ['train', 'eval']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(mode)
            _, estimation_loss = model(X, Y)
            losses[k] = estimation_loss.item()
        res[mode] = losses.mean()
    model.train()
    return res


class Head (nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_context_size, max_context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) / C**0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_of_heads))
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
    def __init__(self, n_embedd,  n_head):
        super().__init__()
        head_size = n_embedd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embedd)
        self.ln1 = nn.LayerNorm(n_embedd)
        self.ln2 = nn.LayerNorm(n_embedd)

    def forward(self, x):
        # ! adding residual connection by summing up with inputs
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
        y_pred = self.head(x)

        if y is None:
            loss = None
        else:
            B, M, V = y_pred.shape
            y_pred = y_pred.view(B * M, V)
            y = y.view(B * M)
            loss = nn.functional.cross_entropy(y_pred, y)
        return y_pred, loss

    def generate(self, idx, new_tokens_count):
        for _ in range(new_tokens_count):
            output, _ = self(idx[:, -max_context_size:])
            output = output[:, -1, :]
            probabilities = nn.functional.softmax(output, dim=-1)
            next_idx = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=-1)
            print(decode(next_idx[0].tolist()), end='')


x_batch, y_batch = get_batch('train')
model = BigramLM().to(device)
m = model.to(device)
out, loss = model(x_batch, y_batch)  # batch_size x max_context_size x vocab_size



model.load_state_dict(torch.load('./model_weight.pt', map_location=device))

# Generate result
model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), new_tokens_count=1000)
