# Deep Recurrent Survival Analysis in PyTorch
> This project features a PyTorch implementation of the <a href='https://arxiv.org/pdf/1809.02403.pdf'>Deep Recurrent Survival Analysis</a> model that is intended for use on uncensored sequential data in which the event is known to occur at the last time step for each observation


## Installation

```
$ pip install drsa
```

## Usage

```python
from drsa.functions import event_time_loss, event_rate_loss
from drsa.model import DRSA
import torch
import torch.nn as nn
import torch.optim as optim
```

```python
# generating random data
batch_size, seq_len, n_features = (64, 25, 10)
data = torch.randn(batch_size, seq_len, n_features)

# generating random embedding for each sequence
n_embeddings = 10
embedding_idx = torch.mul(
    torch.ones(batch_size, seq_len, 1),
    torch.randint(low=0, high=n_embeddings, size=(batch_size, 1, 1)),
)

# concatenating embeddings and features
X = torch.cat([embedding_idx, data], dim=-1)
```

```python
# instantiating embedding parameters
embedding_size = 5
embeddings = torch.nn.Embedding(n_embeddings, embedding_size)
```

```python
# instantiating model
model = DRSA(
    n_features=n_features + 1,  # +1 for the embeddings
    hidden_dim=2,
    n_layers=1,
    embeddings=[embeddings],
)

```

```python
# defining training loop
def training_loop(X, optimizer, alpha, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)

        # weighted average of survival analysis losses
        evt_loss = event_time_loss(preds)
        evr_loss = event_rate_loss(preds)
        loss = (alpha * evt_loss) + ((1 - alpha) * evr_loss)

        # updating parameters
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"epoch: {epoch} - loss: {round(loss.item(), 4)}")
```

```python
# running training loop
optimizer = optim.Adam(model.parameters())
training_loop(X, optimizer, alpha=0.5, epochs=101)
```

    epoch: 0 - loss: 12.9956
    epoch: 10 - loss: 12.8334
    epoch: 20 - loss: 12.6803
    epoch: 30 - loss: 12.5363
    epoch: 40 - loss: 12.4008
    epoch: 50 - loss: 12.2729
    epoch: 60 - loss: 12.1517
    epoch: 70 - loss: 12.0363
    epoch: 80 - loss: 11.9261
    epoch: 90 - loss: 11.8204
    epoch: 100 - loss: 11.7186

