######################################################################################
# File: Audio(MP3)-> Audio(Wav)-> Spectrogram Conversion and Music Genre Sorter
#
# Author: npocodes 2017
# Purpose: C490 Deep Learning, Group project
# Description: This program is designed to read in collections of music/audio files
#              and generate 2 directories, the first holds the audio files sorted into
#              folders based on the genre given for the file. The second set holds 
#              spectrogram image files created from each audio file and they also
#              are sorted into matching genre directories. 2 Matching directories,
#              1 with audio files (wav) and 1 with image files (spectrogram).
#######################################################################################
import matplotlib.pyplot as plt
from scipy.io import wavfile #wavfile tools
import pydub #audio conversions
import eyed3 #ID3 tag reading
import numpy as np
from shutil import copyfile #copyfile(src, dst)
from shutil import move #move(src, dst)
import os
import sys

#Suppress warnings from eyed3 about standards
#when we are trying to read the file's ID3 data
eyed3.log.setLevel("ERROR")

#############
# FUNCTIONS #
#############
# Convert wav to spectrogram in Python
# August 2016 Vijaya Kolachalama
# Modified by npocodes 2017
def graph_spectrogram(wav_file, savePath):  
  pydub.AudioSegment.from_mp3(wav_file).export(wav_file + '.wav', format="wav")
  rate, data = get_wav_info(wav_file + '.wav')
  nfft = 256 # Length of the windowing segments
  fs = rate  # Sampling frequency
  pxx, freqs, bins, im = plt.specgram(data, nfft, fs, cmap='hot')  
  plt.axis('off')
  plt.savefig(savePath + '.png', dpi=100, frameon='false', aspect='normal', bbox_inches='tight', pad_inches=0)
  #Spectrogram saved as a .png
  
def get_wav_info(wav_file):
  rate, data = wavfile.read(wav_file)

  #Data must be flattened to 1D
  retdata = np.array(data)
  print retdata.flatten(), '\n'  

  return rate, retdata.flatten()


################
# BEGIN SCRIPT #
################
test = False;
print "\n### Welcome Message, audSpecSort? ###\n"

#First check for audio collection path in cmdline args
if len(sys.argv) > 1:
  #We have some arguments; (1: subjectAudioPath)
  #Attempt to list the directory contents
  try:
    items = os.listdir(sys.argv[1])

  except IOError:
    print "Error reading directory contents!\n"
    sys.exit()
else:
  #Set default testing audio file
  items = 'The_Clockmaker.mp3'
  test = True;

#Tell user what we've found.
if test:
  print "No audio collection provided...\nUsing test file: The_Clockmaker.mp3\n"
else:
  #save path to audio collection
  audiopath = sys.argv[1]

  #List items found
  print "Audio collection located. Found", len(items), "files in collection:\n"  
  for item in items:
    print item
  print '\n'

#Setup basic directory structure for processed collection storage
#4th, check for an existing directory with that genre if it doesn't exist, create it (2versions)
if not os.path.isdir('wavim'):
  #Make main directory
  os.mkdir('wavim')

if not os.path.isdir('wavim/audio'):
  #Make audio directory
  os.mkdir('wavim/audio')

if not os.path.isdir('wavim/spectrogram'):
  #Make spectrogram dir
  os.mkdir('wavim/spectrogram')
    
#Begin conversions and sorting
for item in items:
  if not item.endswith('*.wav'):
    #Read the track data
    track = eyed3.load(audiopath + '/' + item)
  
    genreStr = str(track.tag.genre)

    #Check for the genre directory (audio)
    if not os.path.isdir('wavim/audio/' + genreStr):
      #make audio genre directory
      os.mkdir('wavim/audio/' + genreStr)

    #Check for the genre directory (spectrogram)
    if not os.path.isdir('wavim/spectrogram/' + genreStr):
      os.mkdir('wavim/spectrogram/' + genreStr)

    #Convert mp3 to spectrogram and save file in associated genre dir
    graph_spectrogram(audiopath + '/' + item, 'wavim/spectrogram/' + genreStr +'/' + item)

    #Copy or Move the audio file? Move will save space!
    move(audiopath + '/' + item + '.wav', 'wavim/audio/' + genreStr + '/' + item)
    #copyfile(audiopath + '/' + item + '.wav', 'wavim/audio/' + genreStr +'/' + item)

#EOP
