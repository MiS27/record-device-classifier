import os

import numpy as np
from scipy import signal
from scipy.io import wavfile


class DataUtil:

    def __init__(self, data_dir, spectrogram_dir, positive_label='A'):
        self.height = 129
        self.width = 35
        self.data_dir = data_dir
        self.spectrogram_dir = spectrogram_dir
        data = []
        self.labels = []
        for label in os.listdir(data_dir):
            label_directory = os.path.join(data_dir, label)
            for file in os.listdir(label_directory):
                wav_file = os.path.join(label_directory, file)
                spectrogram = self._get_spectrogram(wav_file, label)
                data.append(spectrogram)
                self.labels.append(int(label == positive_label))
        self.data = np.reshape(data, (len(data), 1, self.height, self.width))

    def get_data(self):
        return np.array(self.data), np.array(self.labels, dtype=np.uint8)

    def _get_spectrogram(self, wav_file, label):
        spectrogram_file = self._get_spectrogram_file_path(wav_file, label)
        if os.path.exists(spectrogram_file):
            spectrogram = np.load(spectrogram_file)
        else:
            spectrogram = self._generate_spectrogram(wav_file)
            self._save_spectrogram(spectrogram, spectrogram_file)
        return spectrogram

    def _get_spectrogram_file_path(self, wav_file, label):
        return os.path.join(self.spectrogram_dir, label, os.path.basename(wav_file)) + '.npy'

    @staticmethod
    def _generate_spectrogram(file):
        sample_rate, samples = wavfile.read(file)
        frequencies, times, spectrogram = signal.spectrogram(samples)
        return spectrogram

    @staticmethod
    def _save_spectrogram(spectrogram, spectrogram_file):
        os.makedirs(os.path.dirname(spectrogram_file), exist_ok=True)
        np.save(spectrogram_file, spectrogram)
