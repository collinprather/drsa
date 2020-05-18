# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_core.ipynb (unless otherwise specified).

__all__ = ['survival_rate', 'event_rate', 'event_time']

# Cell

def survival_rate(ts):
    return torch.prod(1-ts, dim=1)

def event_rate(ts):
    return 1-survival_rate(ts)

def event_time(ts):
    return ts[:, -1, :] * survival_rate(ts[:, :-1, :])