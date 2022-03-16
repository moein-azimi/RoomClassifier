import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = './babbleoriginal/splitted/'
x = [path+item  for item in os.listdir(path) if item.endswith('.wav')] 
# saving the label of the samples
y = [item.split('_')[1] for item in os.listdir(path) if item.endswith('.wav')]
# A dataframe to be built here: 
df = pd.DataFrame(y,columns=["room"])
# Counting the number of samples
df = df.value_counts().rename_axis('Room').reset_index(name='Count')
plt.figure()
plt.barh(df['Room'],df['Count'],color=['black', 'red', 'green', 'blue', 'cyan','gray'])
for index, value in enumerate(df['Count']):
    plt.text(value, index, str(value))
plt.xticks([0,50,100,150,200,260])
plt.xlabel('Count')
#Saving the fig
plt.savefig('Count.png')
plt.close()
