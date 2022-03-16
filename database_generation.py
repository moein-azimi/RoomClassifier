from pydub import AudioSegment 
from pydub.utils import make_chunks 
import os

'''
The first step is to split the samples: The original files can be downloaded from 
http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
'''

path = './babbleoriginal/'
F = [file for file in os.listdir(path)]
path1 = './babbleoriginal/splitted/'
# Check whether the specified path exists or not
isExist = os.path.exists(path1)
if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(path1)

for i in range(0,len(F)):
    myaudio = AudioSegment.from_file(path+F[i], "wav") 
    # The splitted files -chunks- in millisec 
    chunk_length_ms = 2500
    #Make chunks of sec     
    chunks = make_chunks(myaudio,chunk_length_ms) 
    for j, chunk in enumerate(chunks): 
        chunk_name = F[i]+"{0}.wav".format(j) 
        # File names...
        print ("exporting", chunk_name) 
        chunk.export(path1+chunk_name, format="wav") 