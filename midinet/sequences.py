'''
Sequences need to be transformed into input / output prediction pairings.
'''

import numpy as np


class SemiRedundantSequences:
    '''Given a tensor that is [timestep, ...] cut it into two tensors, input and output.
    Each pair is an input [sample, timestep, ...] .. [sample, timestep + n, ...] to 
    output [sample, timestep + n + 1].
    Note we are testing with all zeros to look at the shape of the ouputs.
    >>> s = SemiRedundantSequences(2, 1)
    >>> s.transform(np.zeros((5,)))[0].shape
    (3, 2)
    >>> s.transform(np.zeros((5,)))[1].shape
    (3,)
    >>> s.transform(np.zeros((5,5)))[0].shape
    (3, 2, 5)
    >>> s.transform(np.zeros((5,5)))[1].shape
    (3, 5)
    '''

    def __init__(self, max_len=16, step=1):
        '''[summary]

        Keyword Arguments:
            max_len {int} -- maximum lenght of a single input (default: {16})
            step {int} -- sliding window to step across the time series (default: {1})
        '''

        self.max_len = max_len
        self.step = step

    def transform(self, sequence_tensor):
        '''Turns an input tensor into the (input, output) tensor tuple.

        Arguments:
            sequence_tensor {np.array} -- time series data tensor

        Returns:
            tuple -- (input, output) tensor pairs
        '''

        inputs = []
        outputs = []
        for i in range(0, len(sequence_tensor) - self.max_len, self.step):
            inputs.append(sequence_tensor[i: i + self.max_len])
            outputs.append(sequence_tensor[i + self.max_len])
        return np.stack(inputs), np.stack(outputs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
