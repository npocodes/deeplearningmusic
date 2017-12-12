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

    #DataVersion - files are sorted into 3 subsets train, test
    #and validate. In each of those by genre. (must be sorted)
      # -No audio version...
      #dataset/spect/{train, test, validate}/<genre>/<spectrogram>
      #dataset/audmage/{train, test, validate}/<genre>/<audmage>

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
# them to create a dataset.
#
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
from multiprocessing import Pool
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
Audio = False   #If True the audio files will be sorted.
Spect = False   #If True, we are working with spectrograms.
Audmage = False #If True, we are working with audmages.
GDATA = False   #If True, we are going to generate a dataset.
CREATE = False  #If True, we are going to create images.
VERBOSE = False #If True, we will print out commentary.
COPY = False    #If True, files are copied when sorting.
TEST = False    #If True, stop creating images after just 5.

PathList = []   #List of file paths(audio/image)
TrackList = []  #List of tuples (filePath, genre)
Meta = []       #List of meta data (from csv file)
MetaT= {}       #Dictionary of track names and matching genres
                #Indexed by track name.
                    
NumNodes = 1    #Number of CPU nodes you will be using(non-zero).
NumCores = 4   #Number of CPU cores per node you will be using(non-zero).
                #Number of workers(sub-processes) = NumNodes * NumCores
                #(16 cores on the GPUs)

S = 0           #Number of completed images(per given instance)
                #ie: 5*32Workers = 160 images done. (test)

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
    if VERBOSE:
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
  # 'sorted/audio/<genre>/<image>'
  # 'sorted/spect/<genre>/<image>'
  # 'sorted/audmage/<genre>/<image>'
  if not os.path.isdir("sorted"):
    os.mkdir("sorted")
  
  #Data version?
  # 'dataset/spect/{train, test, validate}/<genre>/<image>'
  # 'dataset/audmage/{train, test, validate}/<genre>/<image>'
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
    # 'dataset/spect/train/<genre>'
    if GDATA:
      if not os.path.isdir("dataset/spect"):
        os.mkdir("dataset/spect")

      if not os.path.isdir("dataset/spect/train"):
        os.mkdir("dataset/spect/train")
      if not os.path.isdir("dataset/spect/train/"+ genre):
        os.mkdir("dataset/spect/train/"+ genre)

      if not os.path.isdir("dataset/spect/test"):
        os.mkdir("dataset/spect/test")
      if not os.path.isdir("dataset/spect/test/"+ genre):
        os.mkdir("dataset/spect/test/"+ genre)

      if not os.path.isdir("dataset/spect/validate"):
        os.mkdir("dataset/spect/validate")
      if not os.path.isdir("dataset/spect/validate/"+ genre):
        os.mkdir("dataset/spect/validate/"+ genre)

  if Audmage:
    #Create audmage dir (sorted ver)
    if not os.path.isdir("sorted/audmage"):
      os.mkdir("sorted/audmage")
    #Create audmage genre dir
    if not os.path.isdir("sorted/audmage/"+ genre):
      os.mkdir("sorted/audmage/"+ genre)
    
    #Create data version?
    # 'dataset/audmage/train/<genre>'
    if GDATA:
      if not os.path.isdir("dataset/audmage"):
        os.mkdir("dataset/audmage")

      if not os.path.isdir("dataset/audmage/train"):
        os.mkdir("dataset/audmage/train")
      if not os.path.isdir("dataset/audmage/train/"+ genre):
        os.mkdir("dataset/audmage/train/"+ genre)

      if not os.path.isdir("dataset/audmage/test"):
        os.mkdir("dataset/audmage/test")
      if not os.path.isdir("dataset/audmage/test/"+ genre):
        os.mkdir("dataset/audmage/test/"+ genre)

      if not os.path.isdir("dataset/audmage/validate"):
        os.mkdir("dataset/audmage/validate")
      if not os.path.isdir("dataset/audmage/validate/"+ genre):
        os.mkdir("dataset/audmage/validate/"+ genre)

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
      if VERBOSE:
        print 'Using tracks.txt for genre matching. (You wanna go fast!)'
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
        try:
          tgenre = MetaT[fileName]
        except KeyError:
          print 'Missing tracks.txt entry for: '+ str(fileName)
          print 'Please remove tracks.txt so it can be regenerated for this set.'
          print '(Requires tracks.csv)'
          sys.exit()

        print 'File: '+ fullFileName +', Genre: '+ tgenre
        #Create genre directories (and or dataset dirs)
        #If they don't already exist
        doDirs(tgenre)

        #add trackPath and genre as tuple to TrackList
        TrackList.append([fpath, tgenre])

    elif Meta:
      #using tracks.csv (Modified from Joseph Kotva's Code)
      print 'Using tracks.csv for genre matching. Warning: Slower than using tracks.txt'
      print 'Please wait.. Searching csv is slow....'
      if VERBOSE:
        print 'I\'ll make you a tracks.txt file after we search the csv file this one time.'
        print 'That way we can use it when making the images.'
      #Loop Meta data
      p = 0
      f = len(PathList)
      #We will iterate csv data only one time.
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
          if fileName == row[0]:
            if row[32] == 'small' or row[32] == 'medium':
              #ensure proper formatting for dirName for the genre
              genre = str(row[40]).replace(' ', '').replace('/', '-').replace(',', '-')
              if VERBOSE:
                print 'Match Found['+ genre +']: '+ fpath

              doDirs(genre) #Create genre directories
              #add trackPath and genre as tuple to TrackList
              TrackList.append([fpath, genre])
              PathList.remove(fpath) #more Speed SCOTTY!!
              break #cur meta row matched move on, more Speed SCOTTY!!
        #END PathList Loop

        #Keep our user company while they wait...lol
        if not VERBOSE:
          if p == int(.99*f):
            print 'Looks..like..we-made-it!!!!!..!!!!...: 99%'
            print 'Still trying to find:'
            print PathList
          elif p == int(.98*f):
            print 'And..............: 98%'
          elif p == int(.94*f):
            print 'Hey! Don\'t touch that... focus man!!: 94%'
          elif p == int(.86*f):
            print '\nStop it!\n'
          elif p == int(.83*f):
            print 'No no no, I said... ^, so forget it right now. 83%'
          elif p == int(.75*f):
            print 'I change my mind, let\'s not imagine it. 75%'
          elif p == int(.70*f):
            print 'Imagine how long creating images would take if you'
            print 'had to loop the csv file for every image...: 70%'
          elif p == int(f/2):
            print 'Keep calm, it\'s almost over now!'
            print 'The speed will be much faster after this. 50%'
          elif p == int(.25*f):
            print 'We have to do this to get accurate genre matches for our files...sorry: 25%.'
        p += 1
        
        #Did we find them all?  
        if len(PathList) == 0:
          break #I've given her all she's got captain!!
      #End Meta Loop
      if VERBOSE:
        print '\npheww(cough)!'
        print 'Glad we won\'t have to do that again...'
        print 'Wait, you are not planning on using more data than this are you?'
        print 'Sigh... moving on..\n'

      #Next time we'll be ready fer em!!
      if VERBOSE:
        print 'Creating tracks.txt file to speed up future runs with this set.'
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
#This function is now multiprocessed
#There is no longer a loop.
def doSpect(trackL=None, saveDir=None):

  #Use global S as counter for saved spects
  global S

  #Do nothing if test complete
  if S >= 5 and TEST:
    return False #Do nothing

  #Do we have a track path and genre?
  if trackL == None:
    print 'Missing Track information: [trackPath, genre]'
    return False
  
  fpath = str(trackL[0])#File path
  genre = str(trackL[1])#Track Genre

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
        print 'Unable to load: ' + fpath + '\nSkipping...'
        #no s increment here because we didn't make the spectrogram!
        return False #Failure
        #continue #restart loop at next index, skip this file

      #Was the audio file somehow loaded yet has no data points?
      if data.size == 0:
        print 'Unable to load: '+ fpath +'\nFile was opened but there was no data! Corrupted?\nSkipping...'
        return False #Failure
        #continue #restart loop at next index, skip this file

      #Some calculations on the audio sample points
      stft = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
      mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
      log_mel = librosa.logamplitude(mel)
    
      #print 'Generating Spectrogram for: '+ fpath
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
      
      S += 1 #Increment counter
      print 'Finished spectrogram('+ str(S) +'): '+ savePath
      if S == 5 and TEST:
        print 'Stopping spectrograms here, spect test done!'
    else:
      #The spectrogram already exists, skip it
      print savePath +' already exists, skipping...'
      if not TEST:
        S += 1 #Keep counting though!
  
  return True
#END doSpect Function


################
# REMAP VALUES #
################
def remap(x, oMin, oMax, nMin, nMax):
  #Check range
  if oMin == oMax:
    print 'Warning: Zero input range'
    return x
  if nMin == nMax:
    print 'Warning Zero output range'
    return x

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

  #Use global S as counter for saved audmages
  global S 

  #Do nothing if test complete
  if S >= 5 and TEST:
    return False #Do nothing

  #Do we have a track path and genre?
  if trackL == None:
    print 'Missing Track information: [trackPath, genre]'
    return False
  
  fpath = str(trackL[0])#File path
  genre = str(trackL[1])#Track Genre
  
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
        return False #skip this file

      #Was the audio file somehow loaded yet has no data points?
      if data.size == 0:
        print 'Unable to load: '+ fpath +'\nFile was opened but there was no data! Corrupted?\nSkipping...'
        return False #skip this file

      #print 're-configuring audio data for image...'
      #Divide each data value by the sampling rate...
      #We need a way to include the sampling rate and
      #this way seems most obvious...
      if sr != 0:
        data = data/sr #numpy will divide by each value...
      
      #Get min and max value in new audio data array
      audLowValue = np.amin(data) #min value in the audio data
      audHighValue = np.amax(data)#max value in the audio data
      #Remap the audio values into pixel values
      newData = remap(data, audLowValue, audHighValue, 0, 255)
      if np.array_equal(newData, data):
        print 'Unable to remap: '+ fpath +'\nFile was opened but the data is all empty or the same! Corrupted?\nSkipping...'
        return False

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
      S += 1 #Increment index
      print 'Finshed audmage('+ str(S) +'): '+ savePath
      
      #Stop message if reached 5 images
      if S == 5 and TEST:
        print 'Stopping audmages here, audmage test done!'

    else:
      #The spectrogram already exists, skip it
      print savePath +' already exists, skipping...'
      if not TEST:
        S += 1 #Count skips too!

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
      newPath = 'dataset/'+ item +'/train/'+ genres[i] +'/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/train/'+ genres[i] +'/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/train/'+genres[i] +'/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'
    
    #copy the p2 files to the dataset train directory
    for track in p2tracks:
      #print 'Train file: ' + track
      trackPath = 'sorted/'+ item +'/'+ genres[i] +'/'+ track 
      newPath = 'dataset/'+ item +'/test/'+ genres[i] +'/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/test/'+ genres[i] +'/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/test/'+ genres[i] +'/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

    #copy the p3 files to the dataset validate directory
    for track in p2tracks:
      #print 'Validate file: ' + track
      trackPath = 'sorted/'+ item +'/'+ genres[i] +'/'+ track 
      newPath = 'dataset/'+ item +'/validate/'+ genres[i] +'/'+ track
      try:
        copyfile(trackPath, newPath)
      except IOError:
        print 'Unable to copy file: '+ trackPath
        print 'To new location: '+ newPath +'\nskipping...'
      
      if item == 'spect' and Audmage:
        trackPath = 'sorted/audmage/'+ genres[i] +'/'+ track
        newPath = 'dataset/audmage/validate/'+ genres[i] +'/'+ track
        try:
          copyfile(trackPath, newPath)
        except IOError:
          print 'Unable to copy file: '+ trackPath
          print 'To new location: '+ newPath +'\nskipping...'

      if item == 'audmage' and Spect:
        trackPath = 'sorted/spect/'+ genres[i] +'/'+ track 
        newPath = 'dataset/spect/validate/'+ genres[i] +'/'+ track
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
          if VERBOSE:
            print 'Copying '+ fullFileName +' to: sorted/audio/'+ genre +"/"+ fullFileName
          copyfile(fpath, "sorted/audio/"+ genre +"/"+ fullFileName)
        else:
          if VERBOSE:
            print 'Moving '+ fullFileName +' to: sorted/audio/'+ genre +"/"+ fullFileName
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
if __name__ == '__main__':
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
    print 'Could not find tracks.txt'
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
        if option.lower() == 'audio' or option.lower() == '-a':
          #User wants to work with audio files
          Audio = True
        elif option.lower() == 'spect' or option.lower() == 'spectrogram' or option.lower() == '-s':
          #User wants to work with spectrograms
          #Warning can't sort audio and images at same time!
          #If audio was set true as well, then audio is sorted
          #and spectrograms are created/sorted.
          Spect = True
        elif option.lower() == 'audmage' or option.lower() == '-m':
          #User wants to work with audmages
          #Warning can't sort audio and images at same time!
          #If audio was set true as well, then audio is sorted
          #and spectrograms are created/sorted.
          Audmage = True
        elif option.lower() == 'dataset' or option.lower() == '-d':
          #User wants to create a dataset if possible
          GDATA = True
        elif option.lower() == 'create' or option.lower() == '-c':
          #User wants to create spects and/or audmages
          CREATE = True
        elif option.lower() == 'verbose' or option.lower() == '-v':
          #User wants us to talk about whats happening.
          VERBOSE = True
        elif option.lower() == 'copy' or option.lower() == '-p':
          #User wishes to copy when sorting
          COPY = True
        elif option.lower() == 'test' or option.lower('-t'):
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
      print 'Found '+ str(len(PathList)) +' Files'

      matchTracks() #Match tracks with genres
      audioSort()   #Sort the audio tracks in genres
      print 'Finished sorting audio files.'
      
      #Are we going to create image files?
      if CREATE:
        #Figure out how many worker processes to spawn
        numWorkers = NumNodes * NumCores
        if Spect:
          #Do Create Spectrograms
          p = Pool(processes=numWorkers) #set number of worker processes
          sresult = p.map(doSpect, TrackList)#Run doSpect on each element of TrackList
          #ie: Split TrackList between NumWorkers and run doSpect on each element
          print "Finished "+ str(sum(map(int, sresult))) + " spectrograms."
                             #^Calculates number of true function returns

        if Audmage:
          #Do Create Audmages
          p = Pool(processes=numWorkers) #set number of worker processes
          aresult = p.map(doAudmage, TrackList)#Run doAudmage on each element of TrackList
          #ie: Split TrackList between NumWorkers and run doAudmage on each element
          print "Finished "+ str(sum(map(int, aresult))) + " audmages."

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
      #How many were found?
      print 'Found '+ str(len(PathList)) +' Files'

      #Figure out how many worker processes to spawn
      NumWorkers = NumNodes * NumCores

      #match up audio files with genres
      matchTracks() 
      
      if Spect:
        #Do Create Spectrograms
        p = Pool(processes=NumWorkers) #set number of worker processes
        sresult = p.map(doSpect, TrackList)#Run doSpect on each element of TrackList
        #ie: Split TrackList between NumWorkers and run doSpect on each element
        print "Finished "+ str(sum(map(int, sresult))) + " spectrograms."

      if Audmage:
        #Do Create Audmages
        p = Pool(processes=NumWorkers) #set number of worker processes
        aresult = p.map(doAudmage, TrackList)#Run doAudmage on each element of TrackList
        #ie: Split TrackList between NumWorkers and run doAudmage on each element
        print "Finished "+ str(sum(map(int, aresult))) + " audmages."

    elif GDATA:
      #We are Not working with audio files!
      #Not Creating or sorting!
      generateSet(.8, .1, .1)#Generate a new dataset

    else:
      #Not working with audio files
      #Not creating images
      #Not generating datasets
      #Must be sorting images!
      doSearch(sys.argv[1])#Find all the images files in the directory
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
