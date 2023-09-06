#TODO: add the chord generation code here
#TODO; add the chroma array dictionary generator here
#TODO: add the label making code here
#TODO: Figure out a call the training/testing sets from here?
import sys
import os
import numpy as np
from encode import Chromagram
from encode import Encoder

class Loader:
    def __init__(self,source_dir, destination_dir):

        self.source_directory = source_dir
        self.destination_directory = destination_dir
        
        self.all_files = self._get_all_files()
        self.labels = self._generate_labels()

    def _get_all_files(self):
        return [os.path.join(root, fname)
                for root, dirs, files in os.walk(self.source_dir)
                for fname in files if fname.endswith('.wav')]

    def _generate_labels(self):
        return [self._parse_filename(fname) for fname in self.all_files]

    def _parse_filename(self, filename):
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        chord_type = parts[1] 
        chord_name = parts[2]
        root_note_name = parts[3]
        return chord_type, chord_name, root_note_name
    
    def save_npy_files(self):
            for wav_path in self.all_files:
                chroma = Chromagram(wav_path)
                chroma_array = chroma.chromagram
                
                relative_path = os.path.relpath(wav_path, self.source_directory)
                npy_filename = os.path.join(self.destination_directory, relative_path.replace('.wav', '.npy'))
                
                os.makedirs(os.path.dirname(npy_filename), exist_ok=True)
                np.save(npy_filename, chroma_array)  

    def create_numerical_labels_for_root(self):
        # Extract root notes    
        root_notes = [root_note for _, root_note, _ in self.labels]
        # Get unique root notes and assign a unique number to each
        unique_root_notes = sorted(list(set(root_notes)))
        root_to_num = {root: i for i, root in enumerate(unique_root_notes)}
        # Convert root notes to numerical labels
        numerical_labels = [root_to_num[root] for root in root_notes]
        return numerical_labels, root_to_num 
    
    def save_labels(self, label_filename):
        y, _ = self.create_numerical_labels_for_root()
        with open(os.path.join(self.destination_directory, label_filename), 'w') as f:
            for label in y:
                f.write(f"{label}\n")