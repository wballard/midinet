'''
Coder/Decoder for MIDI into numpy for use in machine learning. 

Each MIDI message has a direct bit pattern encoding, in a sense treating the
message encoding as a black and white bitmap.
'''

import mido
import numpy as np


class Encoder:
    '''Encode a MIDI file as a vectorized bitmap pattern.
    >>> enc = Encoder("var/data/Dancing Queen - Chorus.midi")
    >>> enc.numpy().shape
    (398, 32)
    '''

    def __init__(self, midifile):
        '''
        
        Arguments:
            midifile {string} -- file name of a source midi
        '''
        self.midifile = mido.MidiFile(midifile)

    def numpy(self):
        '''Numpy encoding of the MIDI content.

        This limits to just the note messages and formats in a bitmap fashion, as 
        one-hot sytle encoding to generate a bit pattern that represents the MIDI file
        as raw as possible.

        Returns:
            np.array -- a two dimensional [message, bits] array with 32 bits per message
        '''
        buffer = []
        for track in self.midifile.tracks:
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    # max timing length of 255 for a single byte of timing
                    timing = (np.unpackbits(bytearray(bytes([min(msg.time, 255)]))))
                    # mido can render bytes for the core MIDI Message that is the note and channel and velocity
                    tone = (np.unpackbits(msg.bin()))
                    # packing into a single bit pattern
                    one_hot = np.concatenate([tone, timing])
                    buffer.append(one_hot)
        return np.vstack(buffer)


class Decoder:
    def __init__(self, miditensor):
        '''
        
        Arguments:
            miditensor {numpy.ndarray} -- MIDI encoded with {Encoder}
        '''
        pass

    def midi(self):
        '''{mido.MidiFile} representation.
        '''
        pass




