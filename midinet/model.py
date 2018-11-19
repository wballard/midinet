'''
Machine learning models.
'''

#
import os
import shutil
from pathlib import Path

import mido
import numpy as np
from keras.layers import GRU, BatchNormalization, Dense, Reshape, Dropout
from keras.models import Sequential, load_model

from midinet import codec, sequences

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MIDISequencifier:
    '''
    Machine learning sequence model that takes a MIDI file, encodes, and
    learns a sequence generation model representation using Keras/Tensorflow.
    >>> model = MIDISequencifier(mido.MidiFile("var/data/Dancing Queen - Chorus.midi"))
    >>> model
    MIDISequencer 19.9921875 seconds, [164640, 164640, 164640] parameters
    >>> model.save("var/scratch/model")  
    >>> readback = MIDISequencifier.load("var/scratch/model")  
    >>> readback
    MIDISequencer 19.9921875 seconds, [164640, 164640, 164640] parameters
    >>> readback.generate("var/scratch/junk.midi")
    '''

    def __init__(self, midifile, hidden_units=128, dropout=0.2, sequence_length=16):
        '''Given a MIDI file, parse, and create a machine learning sequence model.

        Arguments:
            midifile {mido.MidiFile} -- source MIDI used to learn.
            hidden_units {int} -- number of hidden units to use in the model
            dropout {float} -- percentage dropout
            sequence_length {int} -- number of midi message to use as an input sequence window
        '''

        self.midifile = midifile
        self.channels = codec.Encoder(self.midifile).numpy()

        # encode the channels into input / output pairs
        seq = sequences.SemiRedundantSequences()
        self.inputs_and_outputs = list(map(seq.transform, self.channels))

        # there is a model for each channel, but they are similarly built
        def build(input_output):
            inputs, outputs = input_output
            input_shape = inputs.shape[1:]
            output_shape = outputs.shape[-1]
            model = Sequential()
            # initial reshape to have consistent layering
            model.add(Reshape(input_shape, input_shape=input_shape))
            # all of our encoded inputs are bits, so no real need to normalize here --
            # straight to the recurrent network!
            model.add(GRU(hidden_units, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(GRU(hidden_units))
            model.add(Dropout(dropout))
            # this dense is to shape to the number of output bits
            # using sigmoid to get values on the range [0-1]
            model.add(Dense(output_shape, activation='sigmoid'))
            # this is effectively a multi-class classifier to predict a set of bits
            model.compile(loss='binary_crossentropy', optimizer='adam')
            return model
        self.models = list(map(build, self.inputs_and_outputs))

    def __repr__(self):
        return "MIDISequencer {0} seconds, {1} parameters".format(
            self.midifile.length,
            str([model.count_params() for model in self.models])
        )

    def train(self, epochs=1, batch_size=16):
        for i, (model, (inputs, outputs)) in enumerate(zip(self.models, self.inputs_and_outputs)):
            print('Channel {0}'.format(i))
            model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size)

    def save(self, model_file_path):
        '''Save the model out to a directory to allow multi-models.

        Arguments:
            model_file_path {string} -- file path to a directory
        '''

        directory = Path(model_file_path)
        if directory.exists() and directory.is_dir():
            shutil.rmtree(directory)
        elif directory.exists():
            directory.unlink()

        # directory to store the trained models
        directory.mkdir(parents=True)
        # model files, padded with zeros -- MIDI has a limited number of channels
        # but we need to pull the models back in order
        for i, model in enumerate(self.models):
            model.save(str(directory / Path('{:0>2d}.keras'.format(i))))
        # and the source midi
        self.midifile.save(str(directory / Path('source.midi')))

    def generate(self, target_midi_filename, length=500):
        '''[summary]

        Arguments:
            target_midi_filename {string} -- save midi to this file
            length {int} -- number of generated steps
        '''
        # multi channel output buffer
        output = []
        for i, channel in enumerate(self.channels):
            seed = np.random.permutation(channel)
            # look into the model and see the number of sequence steps needed to make a prediction
            seed_size = self.models[i].layers[0].get_config()[
                'batch_input_shape'][1]
            starting_sequence = seed[0:seed_size]
            # now generate a series of steps -- midi commands with a sliding window
            # this buffer will be filled in by iterative prediction
            generated = np.zeros((length+seed_size, seed.shape[-1]))
            generated[0:seed_size] = starting_sequence
            for step in range(length):
                # need to expand dims since the model expects a bath -- this is just a batch of one
                predict_from_slice = np.expand_dims(
                    generated[step:step+seed_size], 0)
                # store the predicted output back in the buffer
                generated[step +
                          seed_size] = self.models[i].predict(predict_from_slice)
            # skip past the seed and save the output
            output.append(generated[seed_size:])
        # now generate midi with a multi-channel tensor
        encoded_midi = np.stack(output)
        midifile = codec.Decoder(encoded_midi).midi()
        midifile.save(target_midi_filename)

    @classmethod
    def load(cls, model_file_path):
        '''[summary]

        Arguments:
            model_file_path {string} -- file path to a directory
        '''
        directory = Path(model_file_path)
        if directory.exists() and directory.is_dir():
            midifile = mido.MidiFile(str(directory / Path('source.midi')))
            models = []
            for model_file in sorted(directory.glob('*.keras')):
                models.append(load_model(str(model_file)))
            model = MIDISequencifier(midifile)
            model.models = models
            return model
        else:
            raise FileNotFoundError()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
