#!/usr/bin/env python3
import librosa
import numpy as np
import sys
import os
import neuro

class Chromagram:
    
    def __init__(self, fn_wav, Fs=44100, N=2048, H=1024, gamma=None, version='CQT', norm='2'):
        self.fn_wav = fn_wav
        self.Fs = Fs
        self.N = N
        self.H = H
        self.gamma = gamma
        self.version = version
        self.norm = norm
        
        self.raw_audio, self.raw_Fs = self._load_audio()
        self.chromagram, self.Fs_X = self._compute_chromagram()
        self.num_frames, self.num_bins, self.time_steps = self.get_num_frames()
     
    #load audio    
    def _load_audio(self):
        return librosa.load(self.fn_wav, sr=self.Fs)
    
    #compute number of frames and number of bins, as well as time steps! Important for encoding parameters.
    def get_num_frames(self):
        
        num_frames = self.chromagram.shape[1]
        num_bins = self.chromagram.shape[0]
        time_steps = round(self.Fs_X**(-1) * 1000, 1)
        
        return num_frames, num_bins, time_steps
    
    #computes the chromagram in two different ways (STFT, CQT). Pick one (pick QCT for now).
    def _compute_chromagram(self):
        x, Fs = self.raw_audio, self.raw_Fs
        x_dur = x.shape[0] / Fs
        
        if self.version == 'STFT':
            X = librosa.stft(x, n_fft=self.N, hop_length=self.H, center=True, pad_mode='constant')
            if self.gamma is not None:
                X = np.log(1 + self.gamma * np.abs(X)**2)
            else:
                X = np.abs(X)**2
            X = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=self.H, n_fft=self.N)
            
        if self.version == 'CQT':
            X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=self.H, norm=None)
            
        else:
            raise ValueError(f"Unknown version: {self.version}")
            
        #Calls normalization function
        if self.norm is not None:
            X = Chromagram.normalize_feature_sequence(X, norm=self.norm)
          
        #Fs_X is the sampling rate of the chromagram  
        Fs_X = Fs / self.H
        
        
        return X, Fs_X
    
    #staticmethod just means its not dependent on the class itself
    #Noralizes the feature sequence
    @staticmethod
    def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
        """Normalizes the columns of a feature sequence
        Notebook: C3/C3S1_FeatureNormalization.ipynb

        Args:
            X (np.ndarray): Feature sequence
            norm (str): The norm to be applied. '1', '2', 'max' or 'z' (Default value = '2')
            threshold (float): An threshold below which the vector ``v`` used instead of normalization
                (Default value = 0.0001)
            v (float): Used instead of normalization below ``threshold``. If None, uses unit vector for given norm
                (Default value = None)

        Returns:
            X_norm (np.ndarray): Normalized feature sequence
        """
        assert norm in ['1', '2', 'max', 'z']

        K, N = X.shape
        X_norm = np.zeros((K, N))

        if norm == '1':
            if v is None:
                v = np.ones(K, dtype=np.float64) / K
            for n in range(N):
                s = np.sum(np.abs(X[:, n]))
                if s > threshold:
                    X_norm[:, n] = X[:, n] / s
                else:
                    X_norm[:, n] = v

        elif norm == '2':
            if v is None:
                v = np.ones(K, dtype=np.float64) / np.sqrt(K)
            for n in range(N):
                s = np.sqrt(np.sum(X[:, n] ** 2))
                if s > threshold:
                    X_norm[:, n] = X[:, n] / s
                else:
                    X_norm[:, n] = v

        elif norm == 'max':
            if v is None:
                v = np.ones(K, dtype=np.float64)
            for n in range(N):
                s = np.max(np.abs(X[:, n]))
                if s > threshold:
                    X_norm[:, n] = X[:, n] / s
                else:
                    X_norm[:, n] = v

        elif norm == 'z':
            if v is None:
                v = np.zeros(K, dtype=np.float64)
            for n in range(N):
                mu = np.sum(X[:, n]) / K
                sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
                if sigma > threshold:
                    X_norm[:, n] = (X[:, n] - mu) / sigma
                else:
                    X_norm[:, n] = v

        return X_norm
    

class Encoder:
    
    def __init__(self, fn_wav):
        
        chromagram = Chromagram(fn_wav)                         # chromagram object for us to call on
        
        self.fn_wav = fn_wav                                    # filename of wav file
        self.num_frames = chromagram.num_frames                 # number of frames in chromagram, used to calculate interval      
        self.num_bins = chromagram.num_bins                     # number of bins in chromagram, used to calculate dmin and dmax
        self.time_steps = chromagram.time_steps                 # time steps in chromagram, used to calculate interval  amd subinterval_size
        
        self.chroma_as_list = self._get_chroma()                # chromagram as list
        
        self.encoder_params = {
            "dmin": self.num_bins * [0],                        # a list with the length of the number of bins
            "dmax": self.num_bins * [1],
            "encoders": [{ "rate": { "subinterval_size": self.time_steps } }], #if you change stuff, its this line that you're most likely looking for :)
            "interval": self.time_steps * self.num_frames,      # interval is the length of the song in ms
            }
        
        self.spikes = self._encode()                            # encoded chromagram
        
     
    def _get_chroma(self):
        chroma = Chromagram(self.fn_wav).chromagram             # chromagram as numpy array
        return chroma.tolist()

    def _encode(self):
        encoder = neuro.EncoderArray(self.encoder_params)       # encoder object. "neuro.EncoderArray" from framework
        spikes = encoder.get_timeseries_spikes(self.chroma_as_list) # encoded chromagram. "get_timeseries_spikes" from framework
        return spikes    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the audio file as an argument.")
        sys.exit(1)

    # Get the audio file path from the command line argument
    audio_file_path = sys.argv[1]
    
    # Create an Encoder instance
    encoder = Encoder(audio_file_path)
    
    # Get the spikes
    spikes = encoder.spikes
    print(len(spikes))