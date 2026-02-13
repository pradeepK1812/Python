# tiny_gpt.py
import math
import random

# ============================================================
# 1. Autograd Engine (micrograd-style)
# ============================================================

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._children = _children
        self._local_grads = []
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        out._local_grads = [1.0, 1.0]
        return out

    def __radd__(self, other): return self + other

    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        out._local_grads = [-1.0]
        return out

    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        out._local_grads = [other.data, self.data]
        return out

    def __rmul__(self, other): return self * other

    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')
        out._local_grads = [power * (self.data ** (power - 1))]
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        out._local_grads = [1.0 if self.data > 0 else 0.0]
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        out._local_grads = [e]
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        out._local_grads = [1 / self.data]
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ============================================================
# 2. Tiny Dataset (character-level)
# ============================================================

docs = ["emma", "olivia", "ava", "sophia", "isabella"]
uchars = sorted(list(set("".join(docs))))
BOS = len(uchars)
uchars.append("<BOS>")
vocab_size = len(uchars)

# ============================================================
# 3. Model Hyperparameters
# ============================================================

n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print("num params:", len(params))


# ============================================================
# 4. Model Functions
# ============================================================

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs+head_dim]
            kh = [ki[hs:hs+head_dim] for ki in keys[li]]
            vh = [vi[hs:hs+head_dim] for vi in values[li]]

            logits = [sum(qh[j]*kh[t][j] for j in range(head_dim)) /
                      math.sqrt(head_dim) for t in range(len(kh))]
            weights = softmax(logits)
            head = [sum(weights[t]*vh[t][j] for t in range(len(vh)))
                    for j in range(head_dim)]
            x_attn.extend(head)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_res)]

        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_res)]

    return linear(x, state_dict['lm_head'])


# ============================================================
# 5. Training
# ============================================================

learning_rate = 0.01
beta1, beta2 = 0.85, 0.99
eps = 1e-8

m = [0.0]*len(params)
v = [0.0]*len(params)

num_steps = 300

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens)-1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos in range(n):
        logits = gpt(tokens[pos], pos, keys, values)
        probs = softmax(logits)
        loss_t = -probs[tokens[pos+1]].log()
        losses.append(loss_t)

    loss = sum(losses) * (1/n)
    loss.backward()

    lr_t = learning_rate * (1 - step/num_steps)

    for i, p in enumerate(params):
        m[i] = beta1*m[i] + (1-beta1)*p.grad
        v[i] = beta2*v[i] + (1-beta2)*(p.grad**2)
        m_hat = m[i] / (1 - beta1**(step+1))
        v_hat = v[i] / (1 - beta2**(step+1))
        p.data -= lr_t * m_hat / (math.sqrt(v_hat) + eps)
        p.grad = 0

    if step % 50 == 0:
        print(f"step {step:3d} | loss {loss.data:.4f}")


# ============================================================
# 6. Inference
# ============================================================

print("\n--- Samples ---")

for _ in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token = BOS
    out = []

    for pos in range(block_size):
        logits = gpt(token, pos, keys, values)
        probs = softmax(logits)
        token = random.choices(range(vocab_size),
                               weights=[p.data for p in probs])[0]
        if token == BOS:
            break
        out.append(uchars[token])

    print("".join(out))

