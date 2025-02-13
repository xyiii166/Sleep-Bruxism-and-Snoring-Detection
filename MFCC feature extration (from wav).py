import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
file_path = r"C:\Users\lulu\Downloads\human_voice.wav" # Replace with your file path
y, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate

# Extract MFCC features
n_mfcc = 13  # Number of MFCC coefficients
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
print(mfccs)

'''
# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis="time", sr=sr)
plt.colorbar(label="MFCC Coefficients")
plt.title("MFCC Features")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.show()

# Convert to NumPy array
mfccs_array = np.array(mfccs)
print("MFCC Shape:", mfccs_array.shape)  # (n_mfcc, time_frames)
'''