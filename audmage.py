############################################################
# Audmage - Sorts audio collections by genre
#         - Creates and sorts spectrogram images by genre
#         - Creates and sorts audmages by genre
#         - Generates datasets for CNNs using sorted images
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
#   1-  <path to data> : required, audioDir or imgDir (if sorting images)
#   2+- options{audio, spect, audmage, create, dataset, copy, test}
#              (takes 0 or more options)
#
# [Usage Examples]
# ex: ~$ python audmage.py fma_small audio
# only sorts audio by genre, no spectrograms, no audmages, no dataset
# moves audio files, does not copy them
#
#
# ex2: ~$ python audmage.py data/spect  spect
# only sorts spectrograms by genre, notice dir path now for images!
#
#
# ex3: ~$ python audmage.py fma_small create spect dataset
# creates spectrogram images and sorts them, then uses 
# them to create a dataset
#
# ex4: ~$ python audmage.py fma_small
# default, no options does all options!
#
#############################################################

#Import required libs
import matplotlib
matplotlib.use('Agg')
import os    #for file system tools
import sys   #to read command-line arguments 
import numpy as np #matrices and tools
import librosa #spectrogram/audio tools
from librosa import display #Must import seperately

import matplotlib.pyplot as plt #for access to the pyplot module
                                #underlying librosa
from shutil import move #move(src, dst)
from shutil import copyfile #copyfile(src, dst)
from random import shuffle #randomize the dataset selections
import csv

###########
# GLOBALS #
###########
Audio = False   #Flag that we want to work with Audio Files
Spect = False   #Flag that we want to work with Spectrograms
Audmage = False #Flag that we want to work with Audmages
GDATA = False   #Flag that we want to generate a dataset
CREATE = False  #Flag to specify creating, not sorting, images
COPY = False    #Flag to specify copying or moving files when sorting.
TEST = False    #Flag to stop creating images after just 5, for testing.

PathList = []   #List of file paths(audio/image)
TrackList = []  #List of tuples (filePath, genre)
Meta = []       #List of meta data (from csv file)
MetaT= {}       #Dictionary of track names and matching genres
                #Indexed by track name.

####################
# SEARCH FOR FILES #
####################
#Finds all files in all sub-directories of dir given.
#If Audio flag set, only finds .mp3 files
#If no Audio flag set, only finds .png files
def doSearch(dirPath):
  #Is dirPath a directory or file?
  if os.path.isdir(dirPath):
    #Try to read dirPath
    try:
      itemList = os.listdir(dirPath)
    except IOError:
      print 'Failed to open: '+ dirPath
      return False
    #pass each item back into this function
    #and add its results to this one
    for item in itemList:
      doSearch(dirPath +'/'+ item)
  else:
    print 'Found audio file: '+ dirPath
    #dirPath is a file, if looking for audio
    #find only audio, else find only images
    #this avoids picking up files we don't want
    #(what if image file in audio dir?.. we make sure)

    if (Audio or CREATE) and dirPath[-4:] == '.mp3':
      #Only grab .mp3 files and only if working with audio
      PathList.append(dirPath)
    elif (Spect or Audmage) and dirPath[-4:] == '.png':
      #Only grab .png files since we are not working with audio
      PathList.append(dirPath)

  return True
#END doSearch Function


#####################################
# Creating the Directory Structures #
#####################################
def doDirs(genre):
  
  #Sorted version
  if not os.path.isdir("sorted"):
    os.mkdir("sorted")
  
  #Data version?
  if GDATA:
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
    if GDATA:
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
    if GDATA:
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


################
# MATCH TRACKS #
################
#Matches names of files in the PathList global
#with genres read from the tracks.txt, tracks.csv
#or from the audio file itself(if audio files)
#Adds matches as tuple pair to TrackList global
def matchTracks():
  
  #Check for pathList
  if not PathList:
    print 'pathList is empty, no tracks to match. You must first do a search.'
  else:
  
    #Before we start any loops.. where are we finding the genre data?
    #Is tracks.txt available????
    if MetaT:
      print 'Using tracks.txt for genre matching.'
      #Loop each path in pathList
      for fpath in PathList:
        #Split up the path string and get 
        #the file name and extension
        tmp = fpath.split('/') #Split path into list seperating by '/' 
        tmp2 = str(tmp[-1]).split('.') #last element is filename+ext
        fullFileName = tmp[-1]       # filename.mp3
        fileName = str(int(tmp2[0]))  # filename
        fileExt = tmp2[1]             # .mp3/.png
        
        #Get the genre that matches this filename
        tgenre = MetaT[fileName]
          
        print 'File: '+ fullFileName +', Genre: '+ tgenre
        #Create genre directories (and or dataset dirs)
        #If they don't already exist
        doDirs(tgenre)

        #add trackPath and genre as tuple to TrackList
        TrackList.append([fpath, tgenre])

    elif Meta:
      #using tracks.csv (Modified from Joseph Kotva's Code)
      print 'Using tracks.csv for genre matching. Warning: Slower than using tracks.txt'
      print 'A new tracks.txt will be created during this run for faster future runs.'
      #Loop Meta data
      for row in Meta:
        #Loop PathList
        for fpath in PathList:
          #Split up the path string and get 
          #the file name and extension
          tmp = fpath.split('/')
          tmp2 = str(tmp[-1]).split('.')
          fullFileName = tmp[-1] # filename.mp3
          fileName = str(int(tmp2[0]))# filename
          fileExt = tmp2[1] # .mp3/.png
          #Does this track name match the current csv row?
          if fileName == row[0] and row[32] == 'small':
            doDirs(str(row[40])) #Create genre directories
            #add trackPath and genre as tuple to TrackList
            TrackList.append([fpath, str(row[40])])
            PathList.remove(fpath) #more Speed SCOTTY!!
            break #cur meta row matched move on, more Speed SCOTTY!!
        #END PathList Loop
        if len(PathList) == 0:
          break #I've given her all she's got captain!!
      #End Meta Loop
      
      #Next time we'll be ready fer em!!
      print 'Creating tracks.txt file to speed up future runs.'
      #Create new tracks list (tracks.txt)
      tracksFile = open('tracks.txt', 'a')
      for fpath, genre in TrackList:
        #Split up the path string and get 
        #the file name and extension
        tmp = fpath.split('/')
        tmp2 = str(tmp[-1]).split('.')
        fullFileName = tmp[-1] # filename.mp3
        fileName = str(int(tmp2[0])) # filename
        fileExt = tmp2[1]       # .mp3/.png
        print >> tracksFile, fileName, genre
      #End loop
      tracksFile.close()
    else:
      #Skip or read from .mp3 file..
      print 'No genre data, skipping...'
        
  return True
#END matchTracks Function


#######################
# CREATE SPECTROGRAMS #
#######################
def doSpect(trackL=None, saveDir=None):
  #Do we have any track paths?
  if trackL == None and len(TrackList) < 1:
    print 'Missing track information, please do search and match first.\n Or provide list of trackPath/genre pairs'
    return False

  #Global list or given one?
  if trackL == None:
    #User passed no track paths
    sTracks = TrackList
  else:
    if not type(trackL[0]) is tuple:
      #sTracks is not a tuple...
      print 'doSpect() requires track list to be tuple: [trackPath, genre]'
      return False
    else:
      #User gave list of tracks to use.
      sTracks = trackL
  
  print '\nAttempting to generate and sort Spectrograms...\n'
  s = 0 #Counter for spectrograms done
  for fpath,genre in sTracks:
    #Split up the path string and get 
    #the file name and extension
    tmp = fpath.split('/')
    tmp2 = str(tmp[-1]).split('.')
    fullFileName = tmp[-1] # filename.mp3
    fileName = str(int(tmp2[0])) # filename (minus leading zeros)
    fileExt = tmp2[1]       # .mp3/.png

    #Verify the file exists and is accessible
    if not os.path.isfile(fpath):
      #File doesn't exist or isn't accessible.
      print 'File: '+ fullFileName +' does not exist or is not accessible\n'
    else:
      #Create Spectrogram (Modified from Joseph Kotva's Code)
      
      #Setup the save path
      if saveDir == None:
        savePath = 'sorted/spect/'+ genre +'/'+ fileName + '.png'
      else:
        savePath = saveDir +'/'+ fileName + '.png'      

      #Does the spectrogram already exist? Save time, skip it then
      if not os.path.exists(savePath):
        #Try to load the audio file using librosa
        print 'Attempting to load: '+ fpath
        try:
          data, sr = librosa.load(fpath, mono=True) #mono(1channel)
        except IOError:
          print 'Unable to load: ' + fpath
          #no s increment here because we didn't make the spectrogram!
          continue #restart loop at next index, skip this file

        #Was the audio file somehow loaded yet has no data points?
        if data.size == 0:
          print 'Unable to load: '+ fpath +'\nFile was opened but there was no data! Corrupted?\nSkipping...'
          continue #restart loop at next index, skip this file

        #Some calculations on the audio sample points
        stft = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        log_mel = librosa.logamplitude(mel)
      
        print 'Attempting to generate spectrogram image...'
        #Create the spectrogram image
        librosa.display.specshow(log_mel, sr=sr, hop_length=512)
        plt.axis("normal") #axis limits auto scaled to make image sit well in plot box.
        plt.margins(0,0) #remove margins
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) #remove x axis locator
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) #remove y axis locator

        #Save the plotted figure (image) using "SortedVersion" dir structure
        #the image can/will be copied later into a "DataVersion" dir set.
        plt.savefig(savePath, dpi=100, frameon='false', bbox_inches="tight", pad_inches=0.0)
        plt.clf()#Clear the current figure (possibly helps with speed)
        
        s += 1 #Increment counter
        print 'Finished spectrogram('+ str(s) +'): '+ savePath
        if s == 5 and TEST:
          print 'Stopping spectrograms here, spect test done!'
          break
      else:
        #The spectrogram already exists, skip it
        print savePath +' already exists, skipping...'
        if not TEST:
          s += 1 #Keep counting though!

  #END tracks loop
  return True
#END doSpect Function


################
# REMAP VALUES #
################
def remap(x, oMin, oMax, nMin, nMax):
  #Check range
  if oMin == oMax:
    print 'Warning: Zero input range'
    return None
  if nMin == nMax:
    print 'Warning Zero output range'
    return None

  #Check reversed input range
  reverseInput = False
  oldMin = min(oMin, oMax)
  oldMax = max(oMin, oMax)
  if not oldMin == oMin:
    reverseInput = True
    
  #Check reversed output range
  reverseOutput = False
  newMin = min(nMin, nMax)
  newMax = max(nMin, nMax)
  if not newMin == nMin:
    reverseOutput = True

  portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
  if reverseInput:
    portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)
  
  result = portion + newMin
  if reverseOutput:
    result = newMax - portion

  return result
#END remap Function


##################
# CREATE AUMAGES #
##################
#Works in matplotlib v2.0.2
def doAudmage(trackL=None, saveDir=None):
  #Do we have any track paths?
  if trackL == None and len(TrackList) < 1:
    print 'Missing track information, please do search and match first.\n Or provide list of trackPath/genre pairs'
    return False

  #Global list or given one?
  if trackL == None:
    #User passed no track paths
    sTracks = TrackList
  else:
    if not type(trackL[0]) is tuple:
      #sTracks is not a tuple...
      print 'doAudmage() requires track list to be tuple: [trackPath, genre]'
      return False
    else:
      #User gave list of tracks to use.
      sTracks = trackL
  
  print '\nAttempting to generate and sort Audmages...\n'
  a = 0 #Counter for audmages done
  for fpath, genre in sTracks:
    #Split up the path string and get 
    #the file name and extension
    tmp = fpath.split('/')
    tmp2 = str(tmp[-1]).split('.')
    fullFileName = tmp[-1] # filename.mp3
    fileName = str(int(tmp2[0])) # filename (minus leading zeros)
    fileExt = tmp2[1] # .mp3/.png

    #Verify the file exists and is accessible
    if not os.path.isfile(fpath):
      #File doesn't exist or isn't accessible.
      print 'File: '+ fullFileName +' does not exist or is not accessible\n'
    else:
      #Create Audmages!

      #Setup the save path
      if saveDir == None:
        savePath = 'sorted/audmage/'+ genre +'/'+ fileName + '.png'
      else:
        savePath = saveDir +'/'+ fileName + '.png'      

      #Does the audmage already exist? Save time, skip it then
      if not os.path.exists(savePath):      
        #Try to load the audio file using librosa
        print 'Attempting to load: '+ fpath
        try:
          data, sr = librosa.load(fpath, mono=False) #mono(1channel)
        except IOError:
          print 'Unable to load: ' + fpath
          #no s increment here because we didn't make the audmage!
          continue #restart loop at next index, skip this file

        #Was the audio file somehow loaded yet has no data points?
        if data.size == 0:
          print 'Unable to load: '+ fpath +'\nFile was opened but there was no data! Corrupted?\nSkipping...'
          continue #restart loop at next index, skip this file
        
        print 're-configuring audio data for image...'
        #Divide each data value by the sampling rate...
        #We need a way to include the sampling rate and
        #this way seems most obvious...
        newData = data/sr #numpy will divide by each value...

        #Remap the audio values into pixel values
        #Get min and max value in new audio data array
        audLowValue = np.amin(newData) #min value in the audio data
        audHighValue = np.amax(newData)#max value in the audio data
        #audDifValue = (audHighValue - audLowValue) #difference between max and min of audio data(oldRange)
        #pixDifValue = (255 - 0) #difference between max and min of pixel values(newRange)
        newData = remap(newData, audLowValue, audHighValue, 0, 255)

        #resize the matrix
        tmp3 = newData.shape #Read current shape(2,?)
        #print 'Shape ', tmp3
        try:
          valueCount = (tmp3[0]*tmp3[1]) # 2*?
        except IndexError:
          valueCount = (tmp3[0] * 2)#Must be mono file(copy same data to 2nd channel)
          newData = np.vstack((newData,newData))
          #print 'OldShape: ', str(tmp3), 'NewShape: ', str(newData.shape)

        #split the data up into 3 or 4 image channels (RGB/A)
        L = W = int((valueCount/3)**0.5) + 2 #adding 2 to square root to ensure all elements fit (ex: 500x500 img~)
        newData = np.sort(newData, axis=1)  #resort along the 1st axis (try sorting after reshape*)
        #newData = np.flip(newData, axis=0) #flip high>low values (try after reshape*)
        newData.resize(L, W, 3) #reshape/size the matrices to an image size 3-4 channels

        #At this point we have averaged all values by the samplng rate
        #and "remapped" the values to pixel value range and reshaped.
        #All the values can now be treated as pixel values

        plt.axis('normal')
        plt.imsave(savePath, newData, cmap='hot', format='png', dpi=100)
        #Saving as an image lets us store the changes to the numpy matrix
        #for later use, but this can be done on-the-fly without the image conversion.
        #by just "normalizing/scaling" the data values with the sampling rate etc...
        #im.save(newdata, 'sorted/audmages/'+ genre +'/'+ audFileName +'.png')
        a += 1 #Increment index
        print 'Finished audmage('+ str(a) +'): '+ savePath

        if a == 5 and TEST:
          print 'Stopping audmages here, audmage test done!'
          break
      else:
        #The spectrogram already exists, skip it
        print savePath +' already exists, skipping...'
        if not TEST:
          a += 1 #Count skips too!

  #END tracks loop

  return True
#END doAudmage Function


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

    shuffle(tracks) #mix up the tracks to scramble datasets each time
    #all of these tracks belong to same genre so it doesn't effect the splits
    
    #split up the tracks in this genre for each dataset item (train/test/validate)
    p1tracks = tracks[:int(gCount[i]*p1)] #only take p1 percent of these tracks
    p2tracks = tracks[int(gCount[i]*p1):int((gCount[i]*p1)+(gCount[i]*p2))] #take p2 percent of these tracks
    p3tracks = tracks[int((gCount[i]*p1)+(gCount[i]*p2)):] #all the rest... p3 percent of these tracks..

    print 'There are '+ str(len(p1tracks)) +' '+ genres[i] +' tracks for train set.'
    print 'There are '+ str(len(p2tracks)) +' '+ genres[i] +' tracks for test set.'
    print 'There are '+ str(len(p3tracks)) +' '+ genres[i] +' tracks for validate set.'
    print 'Sorting.. please wait..\n'
    #copy the p1 files to the dataset test directory
    for track in p1tracks:
      doDirs(genres[i])#make sure the dir struct is in place
      #print 'Test file: ' + track
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
    
    #copy the p2 files to the dataset train directory
    for track in p2tracks:
      #print 'Train file: ' + track
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

    #copy the p3 files to the dataset validate directory
    for track in p2tracks:
      #print 'Validate file: ' + track
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
  print "Dataset generated using split:  80%-train | 10%-test | 10%-validation."
  return True
#End generateSet function


#AUDIO SORT
def audioSort():
  #Required Data: list of track paths and 
  #list of genres. We need the trackList
  if len(TrackList) < 1:
    #No tracks in track list, can't sort.
    print 'Missing track information, you need to first search and match genres.'
    return False
  else:
    #We have track list
    print 'Attempting to sort Audio files...'
    for fpath, genre in TrackList:
      #Split up the path string and get 
      #the file name and extension
      tmp = fpath.split('/')
      tmp2 = str(tmp[-1]).split('.')
      fullFileName = tmp[-1] # filename.mp3
      fileName = str(int(tmp2[0])) # filename (minus leading zeros)
      fileExt = tmp2[1]       # .mp3/.png
      
      #Is it already there, (from previous run cut short?)
      if not os.path.exists("sorted/audio/"+ genre +"/"+ fullFileName):
        #Move or Copy?
        if COPY:
          copyfile(fpath, "sorted/audio/"+ genre +"/"+ fullFileName)
        else:
          move(fpath, "sorted/audio/"+ genre +"/"+ fullFileName)
      else:
        print 'Audio file: '+ fullFileName  +' is already there, skipping...'
    #end track loop
  return True
#END audioSort Function


#IMAGE SORT
#TrackList should be paths to images not audio files
def imageSort():
  #Required Data: list of track paths, 
  #list of genres. We need the trackList
  if len(TrackList) < 1:
    #No tracks in track list, can't sort.
    print 'Missing track information, you need to first search and match genres.'
    return False
  else:
    #We have track list
    for fpath, genre in TrackList:
      #Split up the path string and get 
      #the file name and extension
      tmp = fpath.split('/')
      tmp2 = str(tmp[-1]).split('.')
      fullFileName = tmp[-1] # filename.mp3
      fileName = str(int(tmp2[0])) # filename (minus leading zeros)
      fileExt = tmp2[1]       # .mp3/.png
      
      #which kind of images?
      if Spect:
        item = 'spect'
      else:
        item = 'audmage'
 
      #Is it already there, (from previous run cut short?)
      if not os.path.exists("sorted/"+ item +"/"+ genre +"/"+ fullFileName):
        #Move or Copy?
        if COPY:
          copyfile(fpath, "sorted/"+ item +"/"+ genre +"/"+ fullFileName)
        else:
          move(fpath, "sorted/"+ item +"/"+ genre +"/"+ fullFileName)
    #end track loop
  return True
#END imageSort Function


#OK All functions are ready, lets begin.

#First try and read the tracks.txt file.
#This file greatly speeds everything up.
#It's used to build a dictionary lookup
#for genre using trackname as a key
try:
  with open('tracks.txt', 'r') as f:
    tmp = list(f.read().splitlines())
  for row in tmp:
    tmp2 = row.split(' ')
    MetaT[tmp2[0]] = tmp2[1]
except IOError:
  print 'Could not read tracks.txt'
  #Try to open and read the tracks.csv file
  #This file is slower do to extra looping.
  try:
    iFile = open("fma_metadata/tracks.csv", "rb")
    Meta = csv.reader(iFile)
  except IOError:
    try:
      iFile = open("tracks.csv", "rb")
      Meta = csv.reader(iFile)
    except IOError:
      print 'Unable to open: fma_metadata/tracks.csv'


#Begin setting up options and inputs #
#Do we have any commandline arguments?
if len(sys.argv) > 1:

  #1 - the path to the audio collection
  if not os.path.isdir(sys.argv[1]) == True:
    print "Error, " + str(sys.argv[1]) +" is not a directory or doesn't exist\n"
    sys.exit()

  #2+ - check for options
  if len(sys.argv) > 2:
    #split up command arguments taking all arguments after arg 1
    #and loop through each option to check which it is.
    for option in sys.argv[2:]:
      if option.lower() == 'audio':
        #User wants to work with audio files
        Audio = True
      elif option.lower() == 'spect' or option == 'spectrogram':
        #User wants to work with spectrograms
        #Warning can't sort audio and images at same time!
        #If audio was set true as well, then audio is sorted
        #and spectrograms are created/sorted.
        Spect = True
      elif option.lower() == 'audmage':
        #User wants to work with audmages
        #Warning can't sort audio and images at same time!
        #If audio was set true as well, then audio is sorted
        #and spectrograms are created/sorted.
        Audmage = True
      elif option.lower() == 'dataset':
        #User wants to create a dataset if possible
        GDATA = True
      elif option.lower() == 'create':
        #User wants to create spects and/or audmages
        CREATE = True
      elif option.lower() == 'copy':
        #User wishes to copy when sorting
        COPY = True
      elif option.lower() == 'test':
        #User wishes to copy when sorting
        TEST = True
      else:
        print 'Unknown option: '+ option +', skipping it...'
    #end option loop
  else:
    #No Options, do all that can be done at one time
    Audio = Spect = Audmage = CREATE = GDATA = True

  #Finished setting up options


  ################
  # SCRIPT BEGIN #
  ################
  #Are we working with audio files?
  if Audio:
    #Do sort audio
    #First search audio dir
    doSearch(sys.argv[1])
    if not PathList:
      print 'Unable to locate audio files in specified directory: '+ sys.argv[1]
      sys.exit()

    matchTracks() #Match tracks with genres
    audioSort()   #Sort the audio tracks in genres
    print 'Finished sorting audio files.'

    if Spect and CREATE:
      #Do Create Spectrograms
      doSpect()

    if Audmage and CREATE:
      #Do Create Audmages
      doAudmage()

    if GDATA:
      #Do generate dataset
      generateSet(.8, .1, .1)

  elif CREATE:
    #Still working with audio files!
    #Creating images (auto sorted)
    doSearch(sys.argv[1])
    if not PathList:
      print 'Unable to locate audio files in specified directory: '+ sys.argv[1]
      sys.exit()

    matchTracks() #match up audio files with genres
    
    if Spect:
      #Create Spects
      doSpect()

    if Audmage:
      #Create Audmages
      doAudmage()
      
  elif GDATA:
    #We are Not working with audio files!
    #Not Creating or sorting!
    generateSet(.8, .1, .1)#Generate a new dataset

  else:
    #Not working with audio files
    #Not creating images
    #Not generating datasets
    #Must be sorting images!
    doSearch()#Find all the images files in the directory
    if not PathList:
      print 'Unable to locate image files in specified directory: '+ sys.argv[1]
      sys.exit()

    matchTracks()#match track names with genres from tracks.txt

    if Spect and Audmage:
      print 'Can only sort one directory at a time.\nplease choose spects or audmages, not both.'
    else:
      imageSort() #Sort images, function will decide which based on commands given
else:
  #No arguments given, need path data!
  print 'You must provide a directory path, exiting...'
