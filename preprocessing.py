import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.decomposition import PCA
import urllib
import IPython.display
import wave
import contextlib

plt.rcParams['figure.figsize'] = (14,4)

song = 'the_flaming_lips_yoshimi_battles_the_pink_robots_part_1-AzlMeTxVdH8.wav'
with contextlib.closing(wave.open(song,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    length = frames / float(rate)
    print(length)


y, sr = librosa.load(song, offset=(length/2)-5, duration=10)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
mfccs = np.delete(mfccs, 1, 0)

mfccs_delta = librosa.feature.delta(mfccs)
mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

# print(mfccs)
# print(mfccs_delta.shape)
# print(mfcc_delta2.shape)

librosa.display.waveplot(y=y, sr=sr)


IPython.display.Audio(y, rate=sr)

librosa.display.specshow(mfccs, sr=sr, x_axis='time')

mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
mfccs_delta = sklearn.preprocessing.scale(mfccs_delta, axis=1)
mfccs_delta2 = sklearn.preprocessing.scale(mfccs_delta2, axis=1)

print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))

librosa.display.specshow(mfccs, sr=sr, x_axis='time')

mfccs_totals = np.concatenate((mfccs, mfccs_delta, mfccs_delta2)).T
print(mfccs_totals.shape)

pca = PCA(n_components=5)
results = pca.fit(mfccs_totals)
print(results.components_.shape)

input = results.components_
input = input.reshape(1, 285)
