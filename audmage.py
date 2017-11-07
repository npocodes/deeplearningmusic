############################################################
# Audmage - Sorts audio collections by genre
#         - Creates and sorts spectrogram images by genre
#         - Creates and sorts audmages by genre
#         - Generates datasets for CNNs using sorted images
#         - ReTags audio files with tracks.csv meta data
#
# Program is designed to work with the fma_small dataset
# but could be used for any audio collection
#
# Author: github/npocodes and C490 Deep Learning Group
#
    #SortedVersion - this set is like the control set
      #Audio/<genre>/<file>
      #Spect/<genre>/<file>
      #Audmage/<genre>/<file>

    #DataVersion - files are not sorted into genre 
    #(the classifiaction program should do that, if it can!)
      # -No audio version...
      #Spect/{train, test, validate}/<randomGenreMusicFile>
      #Audmage/{train, test, validate}/<randomGenreMusicFile>

# Command inputs: 
#   1-  <path to data> : audioDir or imgDir (if sorting images)
#   2+- options{audio, spect, audmage, create, dataset, retag}
#              (0 or more options)
#
# [Usage Examples]
# ex: ~$ python audmage.py data/music-collection audio
# 
# only sorts audio by genre, no spectrograms, no audmages, 
# no dataset, no retag 
#
#
# ex2: ~$ python audmage.py data/spect  spect
# 
# only sorts spectrograms by genre, notice dir path now for images!
#
#
# ex3: ~$ python audmage.py data/music-collection  spect create dataset
#
# creates spectrogram images and sorts them, then uses 
# them to create dataset
#
# ex4: ~$ python audmage.py data/music-collection
# default, no options does all options!
#
#############################################################

#Import required libs
import eyed3 #to read audio meta-tags
import os    #for file system tools
import sys   #to read command-line arguments 
import numpy as np #matrices and tools
import librosa as rosa #spectrogram/audio tools
from librosa import display
import matplotlib.pyplot as plt #for access to the pyplot module
                                #underlying librosa
from shutil import move #move(src, dst)
from shutil import copyfile #copyfile(src, dst)
from random import shuffle #randomize the dataset selections
import csv

#############################################
# RECURSIVE AUDIO DIRECTORY SEARCH AND SORT #
#############################################
def AudioSearch(audioPath):
  if os.path.isdir(audioPath):
    try:
      items = os.listdir(audioPath)
    except IOError:
      print 'Failed to open '+ audioPath
      return False
    #Run dir again...
    for item in items:
      AudioSearch(audioPath +'/'+ item)
  else:
    #Found file...Make sure its mp3 file
    #(can be updated later for more types)
    if audioPath[-4:] == '.mp3':

      print 'Found audio File: '+ audiopath
      #sys.exit()

      #Get the file name minus all the path data.
      tmp = audioPath.split("/") #Split the path str into array
      tmp2 = tmp[-1].split(".") #split the last element into array
      audFullName = tmp[-1] #full file name ('filename.mp3')      
      audFileName = tmp2[0] #name of the file ('fileName')
      if len(tmp2) > 1:
        audExt = tmp2[1] #the ext of the file ('mp3')
      
      #search for the genre in the meta tracks file
      cgenre = ''
      for row in Meta:
        if row[0] == str(int(audFileName)) and row[32] == 'small':
          cgenre = str(row[40])
          break;
      #End Meta Search
      iFile.seek(0)#Return to beginning of file(readerObj)

      #If we couldn't find the genre data 
      #try reading it from audio file
      if cgenre == '' or ReTag:
        #read the audio files meta data
        track = eyed3.load(audioPath) 
        
        #Suppress warnings from eyed3 about standards
        #when we are trying to read the file's ID3 data
        eyed3.log.setLevel("ERROR")
        
        tmpg = str(track.tag.genre) #get the genre tag from file
        tmpg = tmpg.replace(' ', '_').replace('/', '-').replace(',', '-')
        
        #Should we ReTag the audio file with the right genre?
        if Retag:
          if not cgenre == tmpg and cgenre == '':
            if cgenre == '':
              #use file genre data
              genre = tmpg
            else:
              #use csv genre data
              genre = cgenre
              
              #Retag the audio file
              print genre +' != '+ tmpg
              track.tag.genre = u''+ str(genre)
              track.tag.save(audioPath)
              print 'Genre changed to match csv file data: '+ audioPath
      else:
        #Not ReTagging or found genre from csv...
        genre = cgenre
      
      print 'Found genre: ' + genre
      #sys.exit()

      #Create dir structures
      doDirs(genre)      
      
      #Sorting the audio files?
      if Audio == True:
        #Just move the audio file into the proper genre
        #directory. it should already exist now...
        move(audioPath, "sorted/audio/"+ genre +"/"+ audFullName)
        
        #Since the audio was moved, use the new location
        Fg.append(["sorted/audio/"+ genre +"/"+ audFullName, genre])
      else:
        #Audio is in original location
        Fg.append([audioPath, genre])

  return True
#End Audio Sort Method

#######################
# IMAGE SORT Function #
#######################
def ImageSort(imgPath, switch = False):

  #Figure out what we are dealing with (spect or audmage)
  if switch == True:
    itemLabel = 'Audmage'
    itemName = 'audmage'
  else:
    itemLabel = 'Spectrogam'
    itemName = 'spect'

  #Recursively find images within directory given
  if os.path.isdir(imgPath):
    try:
      items = os.listdir(imgPath)
    except IOError:
      print 'Unable to open image dir: '+ imgPath
      return False
    for item in items:
      ImageSort(imgPath +'/'+ item)
  else:
    #Found file...Make sure its png file
    #(can be updated later for more types)
    if imgPath[-4:] == '.png':
      print 'Found '+ itemLabel +': '+ imgPath
      #sys.exit()

      #Get the file name minus all the path data.
      tmp = imgPath.split("/") #Split the path str into array
      tmp2 = tmp[-1].split(".") #split the last element into array
      imgFullName = tmp[-1] #full file name ('filename.png')      
      imgFileName = tmp2[0] #name of the file ('fileName')
      if len(tmp2) > 1:
        imgExt = tmp2[1] #the ext of the file ('png')

      #search for the genre in the meta tracks file
      genre = ''
      for row in Meta:
        if row[0] == str(int(imgFileName)) and row[32] == 'small':
          genre = str(row[40])
          break;
      #End Meta Search
      iFile.seek(0)#Return to beginning of file(readerObj)

      #Did we find the genre data?
      if genre == '':
        print 'Unable to locate genre: '+ imgPath +'\nSkipping...'
        return False
      else:
        #Genre was located..
        print 'Found genre: ' + genre

        #Create dir structures
        doDirs(genre)

        #Finally move the file
        nPath = 'sorted/'+ itemName +'/'+ genre +'/'+ imgFullName
        try:
          move(imgPath, nPath)
        except IOError:
          print 'Unable to move: '+ imgPath
          print 'To new location: '+ nPath +'\nSkipping...'

        print imgFullName +' moved to: '+ nPath

  return True
#End Image Sort Method

###################################
# Creating the Spectrogram Images #
###################################
def doSpect(Fg):
  print 'Attempting to generate and sort Spectrograms...\n'
  
  n=0 #index counter
  for audFilePath,genre in Fg:
    #Get the file name minus all the path data.
    tmp = audFilePath.split("/") #Split the path str into array
    tmp2 = tmp[-1].split(".") #split the last element into array
    audFileName = tmp2[0] #name of the file ('fileName')
    audExt = tmp2[1] #the ext of the file ('mp3')
    
    #Verify the file exists and is accessible
    if not os.path.isfile(audFilePath):
      #File doesn't exist or isn't accessible.
      print 'File: '+ tmp[-1] +' does not exist or is not accessible\n'
    
    else:
      #Begin creating the spectrogram image
      #taken from Joe's conversion script for librosa
      #modified slightly for better image result
      try:
        data, sr = rosa.load(audFilePath, sr=None, mono=True) #mono(1channel)
      except IOError:
        print 'Unable to load: ' + audFilePath
        n += 1 #increment index
        continue #restart loop at next index, skip this file

      stft = np.abs(rosa.stft(data, n_fft=2048, hop_length=512))
      mel = rosa.feature.melspectrogram(sr=sr, S=stft**2)
      log_mel = rosa.logamplitude(mel)
        
      #Create the spectrogram image
      rosa.display.specshow(log_mel, sr=sr, hop_length=512)
      plt.axis("normal") #axis limits auto scaled to make image sit well in plot box.
      plt.margins(0,0) #remove margins
      plt.gca().xaxis.set_major_locator(plt.NullLocator()) #remove x axis locator
      plt.gca().yaxis.set_major_locator(plt.NullLocator()) #remove y axis locator

      #Save the plotted figure (image) using "SortedVersion" dir structure
      #the image can/will be copied later into a "DataVersion" dir set.
      savePath = 'sorted/spect/'+ genre +'/'+ audFileName + '.png'
      plt.savefig(savePath, dpi=100, frameon='false', bbox_inches="tight", pad_inches=0.0)
        
      n += 1 #Increment index
      print 'Finished spectrogram('+ str(n) +'): '+ savePath
      #sys.exit()
          
  #End doSpect Loop
  return True
#End Spectrogram Creation Method

def doAudmage():
  #Not currently Implemented
  return True
#End Audmage Creation Method

#####################################
# Creating the Directory Structures #
#####################################
def doDirs(genre):
  
  #Sorted version
  if not os.path.isdir("sorted"):
    os.mkdir("sorted")
  
  #Data version?
  if DataF:
    if not os.path.isdir("dataset"):
      os.mkdir("dataset")

  #Check first to see if dir is needed
  if Audio:
    #Create audio dir (sorted ver)
    if not os.path.isdir("sorted/audio"):
      os.mkdir("sorted/audio")
    #Create audio genre dir    
    if not os.path.isdir("sorted/audio/"+ genre):
      os.mkdir("sorted/audio/"+ genre)
    #No audio data version...

  if Spect:
    #Create spect dir (sorted ver)
    if not os.path.isdir("sorted/spect"):
      os.mkdir("sorted/spect")
    #Create spect genre dir
    if not os.path.isdir("sorted/spect/"+ genre):
      os.mkdir("sorted/spect/"+ genre)
    
    #Create data version?
    if DataF:
      if not os.path.isdir("dataset/spect"):
        os.mkdir("dataset/spect")
      if not os.path.isdir("dataset/spect/train"):
        os.mkdir("dataset/spect/train")
      if not os.path.isdir("dataset/spect/test"):
        os.mkdir("dataset/spect/test")
      if not os.path.isdir("dataset/spect/validate"):
        os.mkdir("dataset/spect/validate")

  if Audmage:
    #Create audmage dir (sorted ver)
    if not os.path.isdir("sorted/audmage"):
      os.mkdir("sorted/audmage")
    #Create audmage genre dir
    if not os.path.isdir("sorted/audmage/"+ genre):
      os.mkdir("sorted/audmage/"+ genre)
    
    #Create data version?
    if DataF:
      if not os.path.isdir("dataset/audmage"):
        os.mkdir("dataset/audmage")
      if not os.path.isdir("dataset/audmage/train"):
        os.mkdir("dataset/audmage/train")
      if not os.path.isdir("dataset/audmage/test"):
        os.mkdir("dataset/audmage/test")
      if not os.path.isdir("dataset/audmage/validate"):
        os.mkdir("dataset/audmage/validate")

  return True
#End doDirs function

########################
# Generate the dataset #
########################
def generateSet(p1,p2,p3):
  if not p1+p2+p3 == 1:
    if not (p1+p2+p3)/100 == 1:
      #the data split percentages don't equal 100%..
      print "The data split percentages must equal 1"
      sys.exit()
  
  #which items are we creating a dataset for?
  #not audio files because dataset uses images
  item = 'none'
  if Spect and os.path.isdir('sorted/spect'):
    item = 'spect'
  elif Audmage and os.path.isdir('sorted/audmage'):
    item = 'audmage'
  else:
    print 'No image set was chosen or chosen set not found: sorted/'+ item
    sys.exit()

  #Try to get a list of genres by reading dir names from a sorted directory
  try:
    genres = os.listdir('sorted/'+ item) #list of genres
  except IOError:
    print 'Unable to get list of genres from: sorted/'+ item
    #Try opposite item if using and available...
    if item == 'spect' and Audmage:
      try:
        genres = os.listdir('sorted/audmage')
        item = 'audmage' #change item
      except IOError:
        print 'Unable to get list of genres from: sorted/audmage'
        sys.exit()
    elif item == 'audmage' and Spect:
      try:
        genres = os.listdir('sorted/spect')
        item = 'spect'
      except IOError:
        print 'Unable to get list of genres from: sorted/audmage'
        sys.exit()
  #End exception

  gCount = [] #total images for each genre
  iTotal = 0 #total number of images among all genres
  for genre in genres:
    try:
      #count of files listed within genre
      tmpCount = len(os.listdir('sorted/'+ item +'/'+ genre))
    except IOError:
      tmpCount = 0

    gCount.append(tmpCount) #save count for this genre
    iTotal += tmpCount #add this genre count to total
  
  #At this point "item" value matters.. both or one?
  i = 0
  while i < len(genres):
    try:
      tracks = os.listdir('sorted/'+ item +'/'+ str(genres[i]))
    except IOError:
      print 'Unable to get list of tracks from: sorted/'+ item +'/'+ str(genres[i])
      i += 1 #Increment to next genre
      continue #Skip rest of code, go to next genre

    #shuffle(tracks) #mix up the tracks to scramble datasets each time
    
    #split up the tracks in this genre for each dataset item (train/test/validate)
    p1tracks = tracks[:int(gCount[i]*p1)] #only take p1 percent of these tracks
    p2tracks = tracks[int(gCount[i]*p1):int((gCount[i]*p1)+(gCount[i]*p2))] #take p2 percent of these tracks
    p3tracks = tracks[int((gCount[i]*p1)+(gCount[i]*p2)):] #all the rest... p3 percent of these tracks..

    #copy the p1 files to the dataset test directory
    for track in p1tracks:
      trackPath = 'sorted/'+ item +'/'+ genres[i] +'/'+ track 
      newPath = 'dataset/'+ item +'/test/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/test/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/test/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'
    
    #copy the p2 files to the dataset train directory
    for track in p2tracks:
      trackPath = 'sorted/'+ item +'/'+ genres[i] +'/'+ track 
      newPath = 'dataset/'+ item +'/train/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/train/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/train/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

    #copy the p3 files to the dataset validate directory
    for track in p2tracks:
      trackPath = 'sorted/'+ item +'/'+ genres[i] +'/'+ track 
      newPath = 'dataset/'+ item +'/validate/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/validate/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/validate/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

    i += 1 #increment genre index
  #End While Loop
  
  return True
#End generateSet function

#Setup Global values
Audio = False #Audio sorting flag
Spect = False #Spectrogram sorting flag
Audmage = False #Audmage sorting flag
DataF = False #Flag to generate a data set (Combine with Spect or Audmage flags)
CreateF = False #Flag to create image data (Combine with Spect or Audmage flags)
ReTag = False #Flag to retag audio files with csv genre data

Fg = [] #List to hold audio file paths and matching genre in tuples [path, genre]
Meta = [] #Will hold the csv data

#open and read the tracks csv file
try:
  iFile = open("fma_metadata/tracks.csv", "rb")
  Meta = csv.reader(iFile)
except IOError:
  print 'Unable to open: fma_metadata/tracks.csv'
  sys.exit()

#Begin setting up options and inputs #
#Do we have any commandline arguments?
if len(sys.argv) > 1:

  #1 - the path to the audio collection
  if not os.path.isdir(sys.argv[1]) == True:
    print "Error, " + str(sys.argv[1]) +" is not a directory or doesn't exist\n"
    sys.exit()
  
  #3+ - check for options
  if len(sys.argv) > 2:
    #split up command arguments taking all arguments after aug 1
    #and loop through each option to check which it is.
    for option in sys.argv[2:]:
      if option.lower() == 'audio':
        #User wants to sort the audio files
        Audio = True
      elif option.lower() == 'spect' or option == 'spectrogram':
        #User wants to create/sort spectrogram images
        Spect = True
      elif option.lower() == 'audmage':
        #User wants to create/sort audmages
        Audmage = False #disabled
      elif option.lower() == 'retag':
        #User wants to retag the audio files
        ReTag = True
      elif option.lower() == 'dataset':
        #User wants to create a dataset if possible
        DataF = True
      elif option.lower() == 'create':
        #User wants to create spects and/or audmages
        CreateF = True
    #end option loop
  else:
    #No options found, default is all options
    Audio = Spect = DataF = ReTag = True #Audmage = false for now
  
  #Finished setting up options
  #Lets begin with image sorting

  if Spect and not CreateF and not DataF:
    #User has given spect command, this means we SORT spectrograms
    #this also means that argv[1] is path to spectrogram directory not audio
    ImageSort(sys.argv[1], False)
    #At this point the spectrograms have been sorted by genre
    print 'Audmages have been sorted successfully!'
    print 'Please run again with dataset command to create dataset.'
    if not Audmage:
      sys.exit()#Close the program

  if Audmage and not CreateF and not DataF:
    #User has given audmage command, this means we SORT audmages
    #this also means that argv[1] is path to audmage directory not audio
    ImageSort(sys.argv[1], True)
    #At this point the audmages have been sorted by genre
    print 'Audmages have been sorted successfully!'
    print 'Please run again with dataset command to create dataset.'
    sys.exit()#Close the program

### All the commands below require Audio search and so sys.argv[1] is audioPath
###Except DataF it can be used either way

  if Audio or ReTag:
    #User wants to sort and/or ReTag audio files
    AudioSearch(sys.argv[1])
    #At this point we should have located all audio files(Fg ready)
    #If Audio command given, files have also been sorted by genre.
    #If ReTag command given, files have also been retagged.
  
  if CreateF:
    #User wants to create and sort images, but what kind?
    #First though we need to search for audio files to
    #create images out of...if we haven't yet
    if not Audio:
      AudioSearch(sys.argv[1])
      #At this point we have located all audio 
      #files but not sorted them. (Fg ready)

    if Spect:
      #Create/sort spects, (requires Fg)
      doSpect(Fg)

    if Audmage:
      #Create/sort audmages, (requires Fg)
      doAudmage(Fg) #disabled

  if DataF:
    #User wants to generate a dataset, but which kind?
    if Spect:
      #Create Spect dataset
      generateSet(.8,.1,.1)

    if Audmage:
      #Create Audmage dataset
      generateSet(.8,.1,.1)
else:
  #No command line arguments provided, need data collection path!
  print "Error, no path to audio files provided! \nExiting...\n"
  sys.exit()
