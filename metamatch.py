# This script is designed to read csv file that contain metadata on audio files
# and create a new txt file where each line contains a file name and its matching
# genre string. The point being to speed up genre search times and reduce file size.
# for the audmage.py program. This script's speed will increase the longer it runs.
#
# Author: github/npocodes and C490 Deep Learning Group
#

#Import the libraries we are using
import csv
import os
import sys

#Some Global Variables
Fn = [] #List of filenames found in audio dir 
Fg = [] #List of filenames, genre for audio files found
dirpath = 'fma_small' #path to root of audio collection "fma_small"
metapath = 'fma_metadata/tracks.csv' #path to tracks.csv file

#open and read the csv file
ifile = open(metapath)
meta = csv.reader(ifile)

#Find the audio files
for root, dirs, files in os.walk(dirpath):
  for audioF in files:
    if audioF[-4:] == '.mp3':
      #for each audio file
      print 'Found Audio File: '+ str(audioF[:-4])
      Fn.append(str(int(audioF[:-4])))
#End audio search Loop

      
#find genres in the csv file
#that match our file names
print 'Matching genres to file names'
for row in meta:
  for f in Fn:
    if f == row[0] and row[32] == 'small':
      print 'Genre: '+ str(row[40]) + '\n'
      Fg.append([f, str(row[40])]) #save pair found
      Fn.remove(str(f)) #shorten the list, make faster
      break #Stop looking, move to next row
  #End fileName loop
  if len(Fn) == 0:
    break #Stop looking through the meta file
#End meta row loop    
#ifile.seek(0)#reset reader obj pos
#no seek needed, we only look through once

print 'Creating tracks.txt file'
#Create new tracks list (tracks.txt)
tracksFile = open('tracks.txt', 'a')
for fName, genre in Fg:
    print >> tracksFile, fName, genre
#End loop
tracksFile.close()
