import numpy as np

def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice

# fajie
def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # probs = a[idx]
    # probs = probs / np.sum(probs)
    # choice = np.random.choice(idx, p=probs)
    return idx

print sample_top_k(np.array([0.02,0.01,0.01,0.16,0.8]),3)


