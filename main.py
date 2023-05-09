############################################################################################
#libraries:
############################################################################################

from pyAudioAnalysis import audioBasicIO # for readAudioFile()
from pyAudioAnalysis import ShortTermFeatures # for feature_extraction()
import pickle as pkl
from smooth import smooth
from pydub import AudioSegment
import os.path
import warnings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statistics as stat
from sklearn.preprocessing import MinMaxScaler, maxabs_scale
import csv
from scipy.io import wavfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
#############################################################################################
#functions:
#########################################################################################

def get_files(path):
    '''this function gets a  path, reads filenames from path and returns names of relevant files'''
    fn = os.listdir(path)
    fNames = []
    for i in fn:
        if i[-4:] == ".wav" and 'stem' not in i.lower() and 'cerclage' not in i.lower():
            fNames.append(i)
    return fNames

def parse_fname(fname):
    '''this function splits name and takes metadata from it'''
    l = fname.split("-", 6)
    d = {}
    d['fileIndex'] = int(l[0])
    d['date'] = l[1]
    d['surgeon'] = l[2]
    d['operation'] = l[3][0]
    d['stakeIndex'] = int(l[3][1:])
    if len(l) > 5:
        d['stakeNum'] = int(l[4].split('.')[0])
        d['exceptions'] = int(l[5][:-4])
    else:
        d['stakeNum'] = int(l[4][:-4])
        d['exceptions'] = None
    return d

#window calculation function
def win_calc(smoothed_vec, peaks_ind, decay_prec=0.9):
    '''this function calculates a window using STE with decay_prec forward and backward relativly to each peak, deafault: decay_prec=0.9 (90%)'''
    stop_ind = []
    start_ind = []
    # energy calculation for smoothed_vec
    energy_vec = np.power(smoothed_vec, 2)  # an array containing the energy of the smoothed vec
    peaks_energy = energy_vec[peaks_ind]  # an array containing the energy of every peak of the smoothed_vec
    sr = 192000
    w = int(0.14 * sr)  # window of 140ms
    # forward decay section
    for j in range(len(peaks_ind)):
        thrs = []
        thrs = (1 - decay_prec) * energy_vec[peaks_ind[j]]
        if peaks_ind[j] < peaks_ind[-1]:
            search_win = []
            try:
                search_win = energy_vec[peaks_ind[j]:peaks_ind[j + 1]]
                ind = np.where(search_win < thrs)[0][0]
                stop_ind.append(peaks_ind[j] + max(w, ind))
            except Exception as ex:
                stop_ind.append(peaks_ind[j + 1])
        else:
            try:
                search_win = energy_vec[peaks_ind[j]:len(energy_vec)]
                ind = np.where(search_win < thrs)[0][0]
                stop_ind.append(peaks_ind[j] + max(w, ind))
            except:
                stop_ind.append(len(energy_vec) - 1)

    energy_vec_flipped = np.flip(energy_vec)
    for j in range(len(peaks_ind)):
        thrs = []
        thrs = (1 - decay_prec) * energy_vec[peaks_ind[j]]
        if peaks_ind[j] > peaks_ind[0]:
            search_win = []
            try:
                search_win = np.flip(energy_vec[peaks_ind[j - 1]:peaks_ind[j]])
                ind = np.where(search_win < thrs)[0][0]
                start_ind.append(peaks_ind[j] - ind)
            except:
                start_ind.append(peaks_ind[j - 1])
        else:
            try:
                search_win = np.flip(energy_vec[0:peaks_ind[j]])
                ind = np.where(search_win < thrs)[0][0]
                start_ind.append(peaks_ind[j] - ind)
            except:
                start_ind.append(0)

    return start_ind, stop_ind


def read_and_parse_file(path, fname):
    '''this function reads an audio file into the DF'''
    # create a temporary DF that will be returned from function
    temp = pd.DataFrame()
    # extract metadata from file name
    parsed_file = parse_fname(fname)
    # load general recording labels from recording name:
    wav = AudioSegment.from_file(path + '\\' + fname, 'wav', codec='pcm_f64le')
    sr = wav.frame_rate  # get sample rate from wav file
    dt = 1 / sr  # delta time calculated from sr
    # get data vector in list
    sound_vec = list(np.frombuffer(wav.get_array_of_samples(), dtype=np.int32))

    # PRE-PROCESSING PIPELINE (not including spectral analysis)
    # 1. DC offset reduction -
    sound_vec_mean = stat.mean(sound_vec)
    sound_vec = [val - sound_vec_mean for val in sound_vec]

    # 2. Pre-emphasis filter -
    k = 0.9  # parameter of the pre-emphasis filter

    # for i in range(len(sound_vec)-1):
    #   sound_vec[i+1] = sound_vec[i+1] - k*sound_vec[i]

    # 3. Normalization -
    sound_vec = maxabs_scale(sound_vec)

    # 4. smoothing and segmentation
    smoothed_vec = smooth(np.abs(sound_vec), window_len=501)[250:250 + len(sound_vec)]
    smoothed_vec = maxabs_scale(smoothed_vec)
    # find peaks of smoothed sound_vec-
    w = int(0.10 * sr)  # window of 100ms
    thrs = 0.25 * max(smoothed_vec)  # set threshold at 25% of max. power
    p = find_peaks(smoothed_vec, height=thrs, distance=w / 2)
    cutIndex = p[0]  # centers of peaks
    # SLICING OUT HITS
    # pre = 0.2  # for 20 ms
    # n = w + int(w * pre)
    # cutStart = [(i-int(w*pre)) for i in cutIndex] #vector of beginnings of segments
    # cutStart[0] = max(0, cutStart[0]) # making sure that the start index is greater than 0
    # cutEnd = [(i + int(w)) for i in cutIndex] #vector of end of segments
    # cutEnd[-1] = min(len(sound_vec), cutEnd[-1]) # making sure that the end index is smaller than the length of the signal

    cutStart, cutEnd = win_calc(smoothed_vec, cutIndex, decay_prec=0.99992)
    # print('cutIndex: ' + str(cutIndex))
    # print('cutStart: ' + str(cutStart))
    # print('cutEnd: ' + str(cutEnd))

    plt.figure(figsize=(12, 7))
    # plt.ylim(0,0.4)
    # plt.xlim(210000,320000)
    plt.plot(smoothed_vec, color='b')
    plt.scatter(cutIndex, smoothed_vec[cutIndex], color='y')
    plt.scatter(cutStart, smoothed_vec[cutStart], color='g')
    plt.scatter(cutEnd, smoothed_vec[cutEnd], color='r')

    sr = 192000

    # making sure that there are no overlaps
    for i in range(len(cutStart) - 1):
        cutStart[i + 1] = max(cutStart[i + 1], cutEnd[i])
    # update signal and noise lists
    sound_list = []
    noise_list = []
    thrs_large_noise_len = 0.320
    thrs_small_noise_len = 0.160
    if (cutStart[0] > 0) & (sound_vec[0:cutStart[0]].size != 0):
        if len(sound_vec[0:cutStart[0]]) > thrs_large_noise_len * sr:  # sound must be longer than 0.320 sec
            noise_list.append(sound_vec[0:cutStart[0]])
    for i in range(len(cutStart)):
        if sound_vec[cutStart[i]:cutEnd[i]].size != 0:
            sound_list.append(sound_vec[cutStart[i]:cutEnd[i]])
        if (i + 1) <= (len(cutStart) - 1):
            if cutEnd[i] < cutStart[i + 1]:
                if sound_vec[cutEnd[i]:cutStart[i + 1]].size > thrs_large_noise_len * sr:  # !=0):
                    noise_list.append(sound_vec[cutEnd[i]:cutStart[i + 1]])
        else:
            if cutEnd[-1] < len(sound_vec):
                if sound_vec[cutEnd[-1]:len(sound_vec)].size > thrs_large_noise_len * sr:  # !=0):
                    noise_list.append(sound_vec[cutEnd[-1]:len(sound_vec)])

    # dividing the large noise segments into smaller (ths_large_noise_len) small
    new_noise_list = []
    if len(noise_list) > 0:
        start_len = len(noise_list)
        for i in range(start_len):

            large_noise_seg = noise_list[i]
            if len(large_noise_seg) >= thrs_small_noise_len * sr:  # if the large seg is greater in len from the thrs_seg
                mod = len(large_noise_seg) % (thrs_small_noise_len * sr)
                if mod/(thrs_small_noise_len * sr) > 0.75:
                    num_of_seg = math.ceil(len(large_noise_seg) / (thrs_small_noise_len * sr))
                else:  # modulo is less than 75% (120ms)
                    num_of_seg = int(len(large_noise_seg)/(thrs_small_noise_len * sr))
                for k in range(0, num_of_seg-1):
                    if k == num_of_seg-1:
                        new_noise_list.append(
                            large_noise_seg[int(k * (thrs_small_noise_len * sr)):])
                        # from the end of the last seg until the end of array
                    else:
                        new_noise_list.append(
                            large_noise_seg[int(k*(thrs_small_noise_len * sr)):int(((k+1)*thrs_small_noise_len * sr)-1)])
        noise_list = new_noise_list

    # enter every sound segment and noise segment as row to temp DF
    row = pd.DataFrame(parsed_file, dtype="object", index=[0])
    row['sr'] = sr
    row['dt'] = dt
    for n in range(len(sound_list)):
        row['signal'] = row.apply(lambda x: [np.array(x)], axis=1).apply(lambda x: np.array(sound_list[n]))
        row['label'] = "sound"
        temp = pd.concat([temp, row])
    for n in range(len(noise_list)):
        row['signal'] = row.apply(lambda x: [np.array(x)], axis=1).apply(lambda x: np.array(noise_list[n]))
        row['label'] = "noise"
        temp = pd.concat([temp, row])
    sound_list = []
    noise_list = []
    # return the temp DF
    temp_copy = temp.copy()
    return (temp_copy)



#####################################
#main:
#####################################
warnings.filterwarnings("ignore")

path = r'C:\Users\python5\PycharmProjects\DenoisingOrtho\db'
#path = r'C:\Users\python5\PycharmProjects\DenoisingOrtho\test_db'
fnames = get_files(path)
sr = 192000
#create empty DF to contain data
label_cols = ['fileIndex', 'date', 'surgeon', 'operation', 'stakeIndex', 'stakeNum', 'exceptions', 'sr', 'dt', 'signal'
    , 'label'] #additional optional fields:  'sound_vec', 'freq_vec', 't_vec'
#db_label = pd.DataFrame(index=fnames, columns=label_cols, dtype="object")
#create dictionary containing sound vectors for each file name
db_hits = pd.DataFrame()
#read and parse files and append
# them to db_hits
for i in range(len(fnames)):
    db_hits = pd.concat([db_hits, read_and_parse_file(path, fnames[i])])
db_hits = db_hits.reset_index(drop=True)
#adding a column of the spectral map
#db_hits['spectrum'] = db_hits['signal'].apply(lambda row: max_pooling(vec=row, bin_size=500))


print('number of sound vectors: '+str(db_hits[db_hits['label'] == 'sound'].shape[0]))
print('number of noise vectors: '+str(db_hits[db_hits['label'] == 'noise'].shape[0]))

only_sound = db_hits[db_hits['label'] == 'sound']
sound_list = []



#########
# mean sound len
num_of_sound_seg = only_sound.shape[0]
only_signal_sound = only_sound['signal']
only_signal_sound = only_signal_sound.reset_index(drop=True)
sound_seg_len = []

for k in range(db_hits[db_hits['label'] == 'sound'].shape[0]): #for num of sound vectors
    sound_seg_len.append(only_signal_sound[k].shape[0])
sound_seg_len = np.array(sound_seg_len)
mean_sound_len = np.mean(sound_seg_len)/sr
print('\nThe mean SOUND len is: ' + str(mean_sound_len*1000)+'[ms]')

#########
# mean noise len
only_noise = db_hits[db_hits['label'] == 'noise']
noise_list = []

num_of_noise_seg = only_noise.shape[0]
only_signal_noise = only_noise['signal']
only_signal_noise = only_signal_sound.reset_index(drop=True)
noise_seg_len = []

for k in range(db_hits[db_hits['label'] == 'noise'].shape[0]):  # for num of sound vectors
    noise_seg_len.append(only_signal_sound[k].shape[0])
noise_seg_len = np.array(noise_seg_len)
mean_noise_len = np.mean(noise_seg_len)/sr
print('The mean NOISE len is: ' + str(mean_noise_len*1000)+'[ms]')


#######################################################################################################################
# finding the average noise spectrum in order to reduce from the sound spectrum:
#######################################################################################################################
only_noise = db_hits[db_hits['label'] == 'noise']
noise_list = []
noise_spec_power = []
sr = 192000  # sample rate
min_noise_window = int(0.1 * sr)  # minimal window size for noise
max_noise_window = 100000
for n in range(len(only_noise)):
    vec = only_noise['signal'].iloc[n]
    # if len(vec) > max_noise_window:
    #     # separate to number of vectors
    #     num_of_vectors = int(np.ceil(len(vec) / max_noise_window))
    #     for j in range(num_of_vectors):
    #         temp_vec = vec[j * max_noise_window:min((j + 1) * max_noise_window, len(vec) - 1)]
    #         if len(temp_vec) > min_noise_window:
    #             # zero pad to maintain 1 Hz resolution
    #             temp_vec = np.pad(array=temp_vec,
    #                               pad_width=(0, 200000 - len(temp_vec)),
    #                               mode='constant',
    #                               constant_values=(0, 0))
    #             noise_list.append(temp_vec)
    #             # fourier transform
    #             spec_power = np.abs(np.fft.fft(temp_vec)) ** 2
    #             noise_spec_power.append(spec_power)
    # else:
        # zero pad
    vec = np.pad(array=vec, pad_width=(0, 200000 - len(vec)), mode='constant', constant_values=(0, 0))
    noise_list.append(vec)
    # fourier transform
    spec_power = np.abs(np.fft.fft(vec)) ** 2
    noise_spec_power.append(spec_power)

# noise_list now contains padded noise vectors with a length of 200000 samples
# noise_spec_power contains the power of the fft of the noise vectors from noise_list
# average all the spectrums
avg_noise_spec_power = noise_spec_power[0]
for n in range(1, len(noise_spec_power)):
    avg_noise_spec_power = np.add(avg_noise_spec_power, noise_spec_power[n])
avg_noise_spec_power = avg_noise_spec_power / len(noise_spec_power)
np.save(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\avg_noise_spec_power.npy', np.array(avg_noise_spec_power))


# create a time axis for the
timestep = 0.000005
freq = np.fft.fftfreq(len(avg_noise_spec_power), d=timestep)

plt.figure(figsize=(15, 7))
plt.plot(freq, avg_noise_spec_power)
plt.xlim(0, 100000)
plt.yscale('symlog')
plt.title('Average Spectrum of Background Noise', fontsize=22)
plt.xlabel('Frequency [Hz]', fontsize=18)
plt.ylabel('Power [dB]', fontsize=18)
plt.savefig(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\average_noise_spectrum.png')

########################################################################################################################
# reduce noise spectrum from sound spectrum:
########################################################################################################################
# first we create a function that gets the avg_noise_spec_power and a segment od noise/sound and reduces avg_noise_spe_power from the segments fourier transform
# the alpha and beta of the filtration should also be included


def filter_noise (avg_noise_spec_power, vec, alpha, beta):
    if len(vec) > 200000:
        vec = vec[0:199999]
    vec = np.pad(array=vec,
                 pad_width=(0, 200000-len(vec)),
                 mode='constant',
                 constant_values=(0, 0))
    sound_spec = np.fft.fft(vec)
    sound_power = np.abs(sound_spec)**2
    sound_phase = np.angle(sound_spec)
    # some constant to not let the value of the subtraction go negative (see paper about noise cancelation)
    #alpha = 5
    #beta = 0.0005
    # subtract noise_power from sound_power
    res = np.subtract(sound_power, alpha*avg_noise_spec_power)
    for i in range(len(res)):
        if res[i] < 0:
            res[i] = beta*avg_noise_spec_power[i]
    # reconstruct back to a complex fft
    res = np.sqrt(res)  # the power is squared, so we take a square root to turn it to an absolute value
    recon = (res*np.cos(sound_phase) + res*np.sin(sound_phase)*1j)
    filtered_signal = np.fft.ifft(recon)
    e_before = np.sum(np.abs(vec)**2)
    e_after = np.sum(np.abs(filtered_signal)**2)
    # returns the KPI of before_energy/after energy
    return (e_after/e_before)


# test the filtration on different values of beta and alpha
beta_range = [0, 0.0005]
alpha_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
only_sound = db_hits[db_hits['label'] == 'sound']
only_noise = db_hits[db_hits['label'] == 'noise']
for beta in beta_range:
    for alpha in alpha_range:
        # create a new directory for the results of the experiment
        dir_name = r'C:\Users\python5\Desktop\proj\alpha and beta experiment\alpha_'+str(alpha)+'_beta_'+str(beta)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        kpi_sound = []
        kpi_noise = []
        for n in range(len(only_sound)):
            sound_vec = only_sound['signal'].iloc[n]
            kpi_sound.append(filter_noise(avg_noise_spec_power, sound_vec, alpha, beta))
        for n in range(len(only_noise)):
            noise_vec = only_noise['signal'].iloc[n]
            kpi_noise.append(filter_noise(avg_noise_spec_power, noise_vec, alpha, beta))

        sound_mean = np.mean(kpi_sound)
        noise_mean = np.mean(kpi_noise)
        sound_std = np.std(kpi_sound)
        noise_std = np.std(kpi_noise)

        print("\nresults for alpha="+str(alpha)+' ,beta='+str(beta)+' :')
        print('filtered sound energy / unfiltered sound energy: '+str(sound_mean))
        print('filtered noise energy / unfiltered noise energy: ' + str(noise_mean))

        plt.figure(figsize=(15, 10))
        plt.hist(kpi_sound, bins=100)
        plt.title('Distribution of Proportion - filtered signal energy/unfiltered signal energy\nMean Value: '+
                  str(sound_mean) + '\nStandard Deviation: '+str(sound_std), fontsize=14)
        plt.xlabel('Filtered signal energy/Unfiltered signal energy')
        plt.ylabel('Number of Segments')
        plt.savefig(dir_name+'/sound_energy_hist.png')

        plt.figure(figsize=(15, 10))
        plt.hist(kpi_noise, bins=100)
        plt.title('Distribution of Proportion - filtered noise energy/unfiltered noise energy\nMean Value: ' +
                  str(noise_mean) + '\nStandard Deviation: ' + str(noise_std), fontsize=14)
        plt.xlabel('Filtered noise energy/Unfiltered noise energy')
        plt.ylabel('Number of Segments')
        plt.savefig(dir_name + '/noise_energy_hist.png')



#
# #####################################
only_sound = db_hits[db_hits['label']=='sound']
 #vectors that show the energy before and after the subtraction
e_before = []
e_after = []
for n in range(len(only_sound)):
    vec = only_sound['signal'].iloc[n]
    #zero pad
    if len(vec)>200000:
        vec = vec[0:199999]
    vec = np.pad(array=vec,
                 pad_width=(0,200000-len(vec)),
                 mode='constant',
                 constant_values=(0, 0))
    sound_spec = np.fft.fft(vec)
    sound_power = np.abs(sound_spec)**2
    sound_phase = np.angle(sound_spec)
    #some constant to not let the value of the subtraction go negative (see paper about noise cancelation)
    alpha = 5
    beta = 0.0005
    #subtract noise_power from sound_power
    res = np.subtract(sound_power, alpha*avg_noise_spec_power)
    for i in range(len(res)):
        if res[i]<0:
            res[i] = beta*avg_noise_spec_power[i]
    #reconstruct back to a complex fft
    res = np.sqrt(res) #the power is squared so we take a square root to turn it to an absolute value
    recon = (res*np.cos(sound_phase) + res*np.sin(sound_phase)*1j)
    filtered_signal = np.fft.ifft(recon)
    e_before.append(np.sum(np.abs(vec)**2))
    e_after.append(np.sum(np.abs(filtered_signal)**2))

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
plt.subplot(211)
plt.plot(vec, label='before filter')
plt.title('example: sound segment before filtering')
plt.subplot(212)
plt.plot(filtered_signal)
plt.title('example: sound segment after filtering')
plt.savefig(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\filtered_sound_example.png')

#######################################################################################################################
#filter the noise itself:
#######################################################################################################################
e_before_noise = []
e_after_noise = []
for n in range(len(noise_list)):
    vec = noise_list[n]
    noise_spec = np.fft.fft(vec)
    noise_power = np.abs(noise_spec)**2
    noise_phase = np.angle(noise_spec)
    #some constant to not let the value of the subtraction go negative (see paper about noise cancelation)
    alpha = 5
    beta = 0.0005
    #subtract noise_power from sound_power
    res = np.subtract(noise_power, alpha*avg_noise_spec_power)
    for i in range(len(res)):
        if res[i]<0:
            res[i] = beta*avg_noise_spec_power[i]
    #reconstruct back to a complex fft
    res = np.sqrt(res) #the power is squared so we take a square root to turn it to an absolute value
    recon = (res*np.cos(noise_phase) + res*np.sin(noise_phase)*1j)
    filtered_noise = np.fft.ifft(recon)
    e_before_noise.append(np.sum(np.abs(vec)**2))
    e_after_noise.append(np.sum(np.abs(filtered_noise)**2))

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
plt.subplot(211)
plt.plot(vec, label='before filter')
plt.title('example: noise segment before filtering')
plt.subplot(212)
plt.plot(filtered_noise)
plt.title('example: noise segment after filtering')
plt.savefig(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\filtered_noise_example.png')

#first kpi is filtered signal energy devided by unfiltered signal energy
#this needs to be as high as possible
kpi1 = np.mean([aft/bef for (aft, bef) in zip(e_after, e_before)])
print("filtered signal energy / unfiltered signal energy: " + str(kpi1))
#second kpi is filtered noise energy devided by unfiltered noise energy
#this needs to be as low as possible
kpi2 = np.mean([aft/bef for (aft, bef) in zip(e_after_noise, e_before_noise)])
print("filtered noise energy / unfiltered signal energy: " + str(kpi2))

np.save(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\sound_energy_before.npy', np.array(e_before))
np.save(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\sound_energy_after.npy', np.array(e_after))
np.save(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\noise_energy_before.npy', np.array(e_before_noise))
np.save(r'C:\Users\python5\PycharmProjects\DenoisingOrtho\plots\results\noise_energy_after.npy', np.array(e_after_noise))





