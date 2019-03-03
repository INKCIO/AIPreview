import pickle
import torch
import gzip
import numpy
from matplotlib import pyplot

with gzip.open("./mnist.pkl.gz", "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

print(x_train[0], y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

m = torch.tensor([[2, 2], [2, 2]])
print(m @ m)

