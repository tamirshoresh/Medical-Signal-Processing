import numpy as np
import matplotlib.pyplot as plt

#sound energy vectors
s_before = np.load(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\sound_energy_before.npy')
s_after = np.load(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\sound_energy_after.npy')
#noise energy vectors
n_before = np.load(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\noise_energy_before.npy')
n_after = np.load(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\noise_energy_after.npy')

kpi_s = [aft/bef for (aft, bef) in zip(s_after, s_before)]
kpi_n = [aft/bef for (aft, bef) in zip(n_after, n_before)]

plt.figure(figsize=(15,10))
plt.hist(kpi_s, bins=100)
plt.title('Distribution of Proportion - filtered signal energy/unfiltered signal energy\nMean Value: '+str(np.mean(kpi_s)), fontsize=16)
plt.xlabel('Filtered signal energy/Unfiltered signal energy')
plt.ylabel('Number of Segments')
plt.show()
#plt.savefig(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\sound_energy_hist.png')

plt.figure(figsize=(15,10))
plt.hist(kpi_n, bins=100)
plt.title('Distribution of Proportion - filtered noise energy/unfiltered noise energy\nMean Value: '+str(np.mean(kpi_n)), fontsize=16)
plt.xlabel('Filtered noise energy/Unfiltered noise energy')
plt.ylabel('Number of Segments')
plt.show()
#plt.savefig(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\noise_energy_hist.png')
