# Overview
MIDINET is a command line program and model to generate synthetic music from MIDI files. Feed it one or more MIDI files to learn, then
generate computerized music!

# Approach
MIDI files are a series of messages, turning notes on and off in one or more channels of output sound, which makes them
similar to strings.

For any given MIDI file, for each channel, we encode the note messages into a multidimensional array.

With this encoding, we train a recurrent network to predict the next message given a previous window. Think of this 
as teaching a neural network to 'hum along'. Once we have this network, given a few random notes, we can generate a new
random song as often as we like. Some of them might even sound good :).

