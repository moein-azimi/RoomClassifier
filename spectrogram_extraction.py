import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# The folder we have saved the splitted samples :
path1 = './babbleoriginal/splitted/'
path2 = './babbleoriginal/spectograms/'
# Check whether the specified path exists or not
isExist = os.path.exists(path2)
if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(path2)
    
x = os.listdir(path1)
#Extracting the spectograms by the use of librosa library:
for i in range(0,len(x)):
    y, sr = librosa.load(path1+x[i], sr=22500)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    out = '%s.png'%path2+x[i]
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure()
    librosa.display.specshow(log_S, sr=sr)
    plt.axis('off')
    plt.savefig(path2+x[i]+'.png', bbox_inches='tight', dpi=300, frameon='false', pad_inches=0.0)
    print ("extracting", x[i]) 
    plt.close()