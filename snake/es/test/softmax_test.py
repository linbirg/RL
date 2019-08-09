import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == "__main__":
    for _ in range(100):
        x = np.random.randn(4).astype(np.float32)
        sft_max = softmax(x)
        a = np.random.choice(range(4), p=sft_max)
        max_i = np.argmax(x)
        print("softmax:", sft_max, "| a:", a, "| max_i", max_i, "| x:", x)
