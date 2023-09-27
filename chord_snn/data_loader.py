import sys
import os
import numpy as np
from encode import Chromagram
from encode import Encoder

"""
DataLoader Class

This class is designed to facilitate the loading and processing of audio data for neural network training. 
It provides functionalities to load `.wav` files from specified directories, extract chromagram data, 
and generate numerical labels based on chord names, root notes, and chord types.

Attributes:
    folders (list): List of specific subfolders to consider for data loading. If not provided, all subfolders are considered.
    source_dir (str): Directory where the original `.wav` files are stored.
    destination_dir (str): Directory where the processed `.npy` chroma files are saved.
    wav_files (list): List of paths to `.wav` files.
    chroma_files (list): List of paths to `.npy` chroma files.
    labels (list): List of labels extracted from the filenames.
    numerical_label_chord (list): Numerical labels representing chord names.
    numerical_label_roots (list): Numerical labels representing root notes.
    numerical_label_types (list): Numerical labels representing chord types.

Methods:
    _get_files(directory, extension): Returns a sorted list of file paths with the specified extension from the given directory.
    _parse_filename(filename): Parses the filename to extract chord type, chord name, and midi note name.
    _get_labels(): Returns labels for each `.wav` file.
    _get_note_without_octave(note): Extracts the note name without the octave information.
    _create_numerical_labels(labels): Converts chord names, root notes, and chord types to numerical labels.
    get_original_labels(one_hot_encoded_labels): Converts one-hot encoded labels back to root notes.

Usage:
    # To load files from all subfolders:
    loader = DataLoader()
    
    # To load files only from 'major_triad' and 'minor_triad' folders:
    loader = DataLoader(folders=['major_triad', 'minor_triad'])
"""
class DataLoader:
    def __init__(self, folders=None):
        #folders is a list of folders to be used (optional)
        self.folders = folders
        
        #directories used
        self.source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/chords")
        self.destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/npy_chroma_chords")

        #lists of files
        self.wav_files = self._get_files(self.source_dir, '.wav')
        self.chroma_files = self._get_files(self.destination_dir, '.npy') #We use this in training
        #labels and numerical labels
        self.labels = self._get_labels()      
        self.numerical_label_chord, self.numerical_label_roots, self.numerical_label_types = self._create_numerical_labels(self.labels) #We use this in training
        
    def _get_files(self, directory, extension):
        return sorted([os.path.join(root, fname)
                    for root, dirs, files in os.walk(directory)
                    if not self.folders or os.path.basename(root) in self.folders
                    for fname in files if fname.endswith(extension)])


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
    def _create_numerical_labels(self, labels):
        # Extract chord name from labels
        chord_names = [chord_name for _, chord_name, _ in self.labels]
        roots = [self._get_note_without_octave(note) for note in chord_names]

        #echo the chord types from labesl
        chord_types = [chord_type for chord_type, _, _ in self.labels]
        
        # Get unique root notes and assign a unique number to each
        unique_chord_names = sorted(list(set(chord_names)))
        unique_roots = sorted(list(set(roots)))
        unique_types = sorted(list(set(chord_types)))
        
        chord_to_num = {root: i for i, root in enumerate(unique_chord_names)}
        root_to_num = {note: i for i, note in enumerate(unique_roots)}
        type_to_num = {chord_type: i for i, chord_type in enumerate(unique_types)}

        self.num_to_chord = {i: root for root, i in chord_to_num.items()}
        self.num_to_root = {i: note for note, i in root_to_num.items()}
        self.num_to_type = {i: chord_type for chord_type, i in type_to_num.items()}

        # Convert root notes to numerical labels
        numerical_label_chord = [chord_to_num[chord] for chord in chord_names]
        numerical_label_roots = [root_to_num[root] for root in roots]
        numerical_label_types = [type_to_num[chord_type] for chord_type in chord_types]
        
        return numerical_label_chord, numerical_label_roots, numerical_label_types    
    
    
    def get_original_labels(self, one_hot_encoded_labels):
        # Convert one-hot encoded arrays back to numerical labels
        numerical_labels = np.argmax(one_hot_encoded_labels, axis=1)
        
        # Convert numerical labels to root notes
        root_notes = [self.num_to_root[num] for num in numerical_labels]

        return root_notes         

if __name__ == "__main__":
    loader = DataLoader()
    
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