#version 1, local memory, original frames
# import librosa
# file_path = r"C:\Users\lulu\Downloads\human_voice.wav" # Replace with your file path
# y, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# print(mfccs)

# Plotting MFCC
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis="time", sr=sr)
# plt.colorbar()
# plt.title("MFCC Features")
# plt.xlabel("Time")
# plt.ylabel("MFCC Coefficients")
# plt.show()
# print("MFCC Shape:", mfccs.shape)  # (Number of MFCCs, Time Frames)


#verison 2, online url, original frame
import requests
import librosa
import io
import soundfile as sf
wav_url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_0.wav"
response = requests.get(wav_url)
if response.status_code == 200:
    y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
else:
    raise Exception("Failed to download the audio file")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(mfccs)




