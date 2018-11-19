'''
Coder/Decoder for MIDI into numpy for use in machine learning. 

Each MIDI message has a direct bit pattern encoding, in a sense treating the
message encoding as a black and white bitmap.
'''

import mido
import numpy as np


class Encoder:
    '''Encode a MIDI file as a vectorized bitmap pattern.
    >>> enc = Encoder(mido.MidiFile("var/data/Dancing Queen - Chorus.midi"))
    >>> enc.numpy().shape
    (3, 252, 32)
    >>> enc.numpy().dtype
    dtype('uint8')
    >>> enc = Encoder(mido.MidiFile("var/data/BRAND1.mid"))
    >>> enc.numpy().shape
    (12, 6328, 32)
    '''

    def __init__(self, midifile):
        '''

        Arguments:
            midifile {mido.MidiFile} -- file object containing midi
        '''
        self.midifile = midifile

    def numpy(self):
        '''Numpy encoding of the MIDI content.

        This limits to just the note messages and formats in a bitmap fashion, as 
        one-hot sytle encoding to generate a bit pattern that represents the MIDI file
        as raw as possible.

        Returns:
            np.array -- a three dimensional [track, message, bits] array with 32 bits per message
        '''
        track_buffer = []
        for i, track in enumerate(self.midifile.tracks):
            buffer = []
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    # max timing length of 255 for a single byte of timing
                    timing = (np.unpackbits(
                        bytearray(bytes([min(msg.time, 255)]))))
                    # mido can render bytes for the core MIDI Message that is the note and channel and velocity
                    tone = (np.unpackbits(msg.bin()))
                    # packing into a single bit pattern
                    one_hot = np.concatenate([tone, timing])
                    buffer.append(one_hot)
            # final pad of zeros, this allows us to handle empty tracks
            if len(buffer) == 0:
                buffer.append(np.zeros(32, dtype=np.uint8))
            track_buffer.append(np.vstack(buffer))
        # this needs to be rectangular, not ragged, so we will be padding
        max_length = max([len(buffer) for buffer in track_buffer])
        padded_buffers = []
        for buffer in track_buffer:
            padded_buffer = np.pad(
                buffer, ((0, max_length-len(buffer)), (0, 0)), 'constant', constant_values=0)
            padded_buffers.append(padded_buffer)
        return np.stack(padded_buffers, axis=0)


class Decoder:
    '''Decode a tensor representation MIDI stream.

    Each input bit may be a floating point and need to be rounded to a hard 1 or 0.
    >>> enc = Encoder(mido.MidiFile("var/data/Dancing Queen - Chorus.midi"))
    >>> dec = Decoder(enc.numpy())
    >>> midi = dec.midi()
    >>> midi.save("var/scratch/test.midi")
    '''

    def __init__(self, miditensor):
        '''

        Arguments:
            miditensor {numpy.ndarray} -- MIDI encoded with {Encoder}
        '''
        self.miditensor = miditensor

    def midi(self, ticks_per_beat=128):
        '''Transforms the encoded midi tensor into a file object which can then be saved.

        Arguments:
            ticks_per_beat {int} -- controls the tempo of the playback, higher is faster

        Returns:
            {mido.MidiFile} representation.
        '''
        # first -- round and decode the message bits
        as_bytes = np.packbits(
            np.round(self.miditensor).astype(np.bool), axis=2)
        # now the actual file build, nothing really special here
        midifile = mido.MidiFile()
        midifile.ticks_per_beat = ticks_per_beat
        # now track and message building
        for track_bytes in as_bytes:
            track = mido.MidiTrack()
            for message_bytes in track_bytes:
                try:
                    # our message format is the first three bytes are pure MIDI, then the fourth byte is time
                    message = mido.Message.from_bytes(message_bytes[0:3])
                    message.time = message_bytes[3]
                    # here is a wee bit of error checking -- it is possible networks generate messages
                    # of invalid types -- so we'll just skip those
                    if message.type in ['note_on', 'note_off']:
                        track.append(message)
                # malformed messages are forgiven, particularly pad messages which are zeros
                except ValueError:
                    pass
            midifile.tracks.append(track)
        return midifile


if __name__ == "__main__":
    import doctest
    doctest.testmod()
