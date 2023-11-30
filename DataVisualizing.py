#Data visualizing
# data link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

#IMPORT THE LIBRARIES
import pandas as pd
import numpy as np
import random
import os
import sys
import shutil

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from tensorflow import layers, models #this is giving an error i will comment it out for now

# to play the audio files
import IPython.display as ipd
from IPython.display import Audio
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Define the path to your dataset directory
sample_dataset_dir = "/content/GoogleDrive/MyDrive/Sample Dataset"
os.listdir(path = sample_dataset_dir) # prints the files in the dataset directory to make sure we got the right path
dataset_dir = '/content/GoogleDrive/MyDrive/Dataset'