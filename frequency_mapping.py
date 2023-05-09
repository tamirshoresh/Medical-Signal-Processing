import numpy as np
import pandas as pd

#######################################################################################################################
#read data from saved pkl file
df = pd.read_pickle(r'C:\Users\python5\Desktop\proj\without_peaks\db_hits_without_peaks.pkl')
df = df[['label', 'spectrum', 'stakeNum']]

sound_spectra = np.zeros((2,1))
print("creating sound vector\n\n\n")
for n in df[df['label'] == 'sound'].index:
    sound_spectra = np.concatenate((sound_spectra, df.iloc[n].spectrum), axis=1)
np.save(r'C:\Users\python5\Desktop\proj\without_peaks\sound_spectra_without_peaks', sound_spectra)

# create array of all noise vectors
noise_spectra = np.zeros((2, 1))
print("creating noise vector\n\n\n")
for n in df[df['label'] == 'noise'].index:
    noise_spectra = np.concatenate((noise_spectra, df.iloc[n].spectrum), axis=1)
np.save(r'C:\Users\python5\Desktop\proj\without_peaks\noise_spectra_without_peaks', noise_spectra)
