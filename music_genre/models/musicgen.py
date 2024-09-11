import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# importing file using filepath
file = r"C:\Users\unmes\Downloads\11- Preprocessing audio data for deep learning_code_blues.00000.wav"



# waveform: full waveform against time of the loaded music file

signal, sr = librosa.load(file, sr=22050)  # sr*T= 22050 * 30
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()



# fft ->> spectrum
# applying fft to convert the wave from time to frequency domain

fft = np.fft.fft(signal) 
magnitude = np.abs(fft) #getting the magnitude of the values 
frequency = np.linspace(0, sr, len(magnitude))
# consider frequency interval between 0 hertz and sample rate itself
# this will create a power spectrum


#commenting out so that we dont have multiple plots in the program
#plot of the whole power spectrum 
plt.plot(frequency, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()





# power spectrum focusing only on half of the total
# as second half/right spectrum is just replica of left
 
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
plt.plot(left_frequency, left_magnitude)
plt.xlabel("LeftFrequency")
plt.ylabel("LeftMagnitude")
plt.show()




#short time fourier transform is needed in order to get 
# how much each frequency contribute to the plot
# stft creates spectrogram which has freq and magni

n_fft= 2048 #number of samples per fft, considering these amount of samples during fft
hop_length = 512 #length of how much to slide to right in the music file after fft and before next fft to calculate


stft = librosa.core.stft(signal, hop_length= hop_length, n_fft=n_fft)
spectrogram= np.abs(stft) #to get absolute values as magnitude


log_spectrogram= librosa.amplitude_to_db(spectrogram)
#using log spectrogram in order to increase the readings/ zoom in using log 

"""
librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
plt.xlabel("time")
plt.ylabel("frequency")
plt.colorbar() # adding color grade Y axes
plt.show()
"""



#MFCC
mfcc = librosa.feature.mfcc(signal,n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length)
plt.xlabel("time")
plt.ylabel("mfcc")
plt.colorbar() # adding color grade Y axes
plt.show()







