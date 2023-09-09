import sys
import os
import numpy as np
from encode import Chromagram
from encode import Encoder

class data_loader:
    def __init__(self):
        #directories used
        self.source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/chords")
        self.destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/npy_chroma_chords")

        #lists of files
        self.wav_files = self._get_files(self.source_dir, '.wav')
        self.chroma_files = self._get_files(self.destination_dir, '.npy') #We use this in training
        #labels and numerical labels
        self.labels = self._get_labels()      
        self.y, self.root_to_num = self.create_numerical_labels(self.labels) #We use this in training
        
    def _get_files(self, directory, extension):
        return [os.path.join(root, fname)
                for root, dirs, files in os.walk(directory)
                for fname in files if fname.endswith(extension)]

    def _parse_filename(self, filename):
        base_name = os.path.basename(filename)
        root_name = os.path.splitext(base_name)[0]
        parts = root_name.split('_')
        chord_type = parts[1] 
        chord_name = parts[2]
        midi_note_name = parts[3]
        return chord_type, chord_name, midi_note_name
    #this is chord type, chord name, and root note name for each file in wav_files
    def _get_labels(self):
        return [self._parse_filename(fname) for fname in self.wav_files]
    
    def _get_note_without_octave(self, note):
        return ''.join([char for char in note if not char.isdigit()])
    
    #this is the directory where the spike data will be saved
    def create_numerical_labels(self, labels):
        # Extract chord name from labels
        chord_names = [chord_name for _, chord_name, _ in self.labels]
        roots = [self._get_note_without_octave(note) for note in chord_names]
        
        # Get unique root notes and assign a unique number to each
        unique_chord_names = sorted(list(set(chord_names)))
        unique_roots = sorted(list(set(roots)))
        
        chord_to_num = {root: i for i, root in enumerate(unique_chord_names)}
        root_to_num = {note: i for i, note in enumerate(unique_roots)}

        # Convert root notes to numerical labels
        numerical_label_chord = [chord_to_num[chord] for chord in chord_names]
        numerical_label_roots = [root_to_num[root] for root in roots]
        
        return numerical_label_chord, numerical_label_roots               

if __name__ == "__main__":
    loader = data_loader()
    
    # Create and save the chromagrams
    for wav_path in loader.wav_files:
        chroma = Chromagram(wav_path)
        chroma_array = chroma.chromagram
        
        # Replicate the directory structure for spike data
        relative_path = os.path.relpath(wav_path, loader.source_dir)
        spike_filename = os.path.join(loader.destination_dir, relative_path.replace('.wav', '.npy'))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(spike_filename), exist_ok=True)
        
        # Save the spike data to disk
        np.save(spike_filename, chroma_array)