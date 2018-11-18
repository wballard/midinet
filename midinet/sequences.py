'''
Sequences need to be transformed into input / output prediction pairings.
'''

import numpy as np

class SemiRedundantSequences:
    '''Given a tensor that is [timestep, ...] cut it into two tensors, input and output.
    Each pair is an input [sample, timestep, ...] .. [sample, timestep + n, ...] to 
    output [sample, timestep + n + 1]
    >>> s = SemiRedundantSequences(2, 1)
    >>> s.transform([1,2,3,4,5])
    (array([[1, 2],
           [2, 3],
           [3, 4]]), array([[3],
           [4],
           [5]]))
    '''
    def __init__(self, max_len=16, step=1):
        self.max_len = max_len
        self.step = step

    def transform(self, sequence_tensor):
        inputs = []
        outputs = []
        for i in range(0, len(sequence_tensor) - self.max_len, self.step):
            inputs.append(sequence_tensor[i: i + self.max_len])
            outputs.append(sequence_tensor[i + self.max_len])
        return np.vstack(inputs), np.vstack(outputs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
