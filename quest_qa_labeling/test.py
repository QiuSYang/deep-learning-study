import torch
import pandas as pd
import numpy as np

class Iter(object):
    def __init__(self):
        self.data = [1,2,3,4,5,6]

    def __iter__(self):
        return iter(self.data)


if __name__ == '__main__':
    q = pd.DataFrame(np.random.rand(5, 2), columns=['rignt', 'left'])
    print(q)
    state_action = q.iloc[3, :]
    print(state_action.max())
