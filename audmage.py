# This program will read audio collections meta-data, create spectrogram image files, 
# and convert audio data into image like matrices that can be used in convultional neural
# networks and save those matrices as images, "audmages". 3 directory structures will be 
# created: 1-audio files, 2-spectrograms, 3-audmages.
#
# Why audmages if spectrograms work in cnns? Spectrograms take alot longer to generate...
# "audmages" may perform better in a runtime conversion situation(on-the-fly) and may
# contain more unique identifying information than spectrograms do. (Maybe...)

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
#   1-<path to data>
#   2-True/False(create dataset)
#   3-options{audio, spect, audmage, retag}(0 or more options)
#
# ex: ~$ python audmage.py data/music-collection False audio
# only sorts audio by genre, no spectrograms, no audmages 
#
# ex2: ~$ python audmage.py data/music-collection True audmages
# only sorts/creates audmages and makes a dataset of audmages
#
# ex3: ~$ python audmage.py data/music-colection
# default, creates/sorts audio,spectrograms, and audmages, no dataset created

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
from PIL import Image

##################################################
# RECURSIVE AUDIO DIRECTORY SEARCH AND SORT LOOP #
##################################################
def audSearch(dirp):
  #dirp = directory path
  if os.path.isdir(dirp):
    #get list of items in this dir
    items = os.listdir(dirp)
    #run each item through this function
    for item in items:
      #passing value as a path...
      audSearch(dirp +'/'+ item)
    #end items loop
  else:
    #dirp was not a directory
    #it must be a file! (or a symlink to a file...)
    #check if it is an mp3 file...? (.mp3, .wav, etc..?)
    if dirp[-4:] == '.mp3': #or dirp[-4:] == '.wav':
      #dirp is probably an audio file
      #lets begin...
      audPath = dirp
      
      #print 'Found audio file: ' + audPath
      #sys.exit()

      #Get the file name minus all the path data.
      tmp = audPath.split("/") #Split the path str into array
      tmp2 = tmp[-1].split(".") #split the last element into array
      audFullName = tmp[-1] #full file name ('filename.mp3')      
      audFileName = tmp2[0] #name of the file ('fileName')
      audExt = tmp2[1] #the ext of the file ('mp3')
      
      #search for the genre in the meta tracks file
      cgenre = ''
      for row in meta:
        if row[0] == audFileName:
          cgenre = row[40]
          break;

      #If we couldn't find the genre data 
      #try reading it from file
      if cgenre == '' or ReTag:
        #read the audio files meta data
        track = eyed3.load(audPath) 
        
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
              track.tag.save(audPath)
              print 'Genre changed to match csv file data: '+ fpath
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
        move(audPath, "sorted/audio/"+ genre +"/"+ audFullName)
        
        #Since the audio was moved, use the new location
        Fg.append(["sorted/audio/"+ genre +"/"+ audFullName, genre])
      else:
        #Audio is in original location
        Fg.append([audPath, genre])

  return True
#End audsearch function

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


###################################
# Creating the Spectrogram Images #
###################################
def doSpect(Fg):
  print 'Attempting to generate Spectrograms...\n'
  n=0
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
      data, sr = rosa.load(audFilePath, sr=None, mono=True) #mono(1channel)?
      
      stft = np.abs(rosa.stft(data, n_fft=2048, hop_length=512))
      mel = rosa.feature.melspectrogram(sr=sr, S=stft**2)
      log_mel = rosa.logamplitude(mel)
      
      #display the image? or create? or both?
      rosa.display.specshow(log_mel, sr=sr, hop_length=512)
      plt.axis("normal")
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())

      #Save the plotted figure (image) using "SortedVersion" dir structure
      #the image can/will be copied later into a "DataVersion" dir set.
      savePath = 'sorted/spect/'+ genre +'/'+ audFileName + '.png'
      plt.savefig(savePath, dpi=100, frameon='false', bbox_inches="tight", pad_inches=0.0)
      
    n += 1
    print 'Finished spectrogram('+ str(n) +'): '+ savePath
    #sys.exit()
  #End doSpect Loop
  return True
#End doSpect function

##################################
# Create audio images "Audmages" #
#       (not spectrograms)       #
##################################
# !! NOT CURRENTLY WORKING
def doAudmage(Fg):

  for audFilePath,genre in Fg:
    #Get the file name minus all the path data.
    tmp = audFilePath.split("/") #Split the path str into array
    tmp2 = tmp[-1].split(".") #split the last element into array
    audFileName = tmp2[0] #name of the file ('fileName')
    audExt = tmp2[1] #the ext of the file ('mp3')

    #Now we need to read the audio data
    #and convert it into a format that is readable by
    #the convultional neural networks (image like)
    data, sr = rosa.load(audFilePath, mono=False)

    #print len(data)
    #print data.shape
    #print data
    #sys.exit()
    
    #data = array of channels and amplitudes (+/-)
    #sr = audio sampling rate

    #Divide each data value by the sampling rate...
    #We need a way to include the sampling rate and
    #this way seems most obvious...
    newdata = data/sr #numpy will divide by each value...

    
    #Get lowest value in new data array
    lowValue = np.amin(newdata)
    print 'lowvalue = ' + str(lowValue)

    #If lowest value is negative...
    if lowValue < 0:
      #convert to positive and add to every value
      #making all values positive.
      newdata += (lowValue * (-1))*(10**7)
    
    print newdata
    newdata = newdata.astype(int)
    print newdata
    newdata = np.resize(newdata, (1152,1152))
    print newdata.shape
    print newdata
    
    im = Image.fromarray(np.uint8(newdata))
    im.show()
    #At this point we have averaged all values by the samplng rate
    #and "normalized" the resulting values to a positive scale.
    #All the values can now be treated as pixel values
    #imshow(newdata)
    sys.exit()

    #Saving as an image lets us store the changes to the numpy matrix
    #for later use, but this can be done on-the-fly without the image conversion.
    #by just "normalizing/scaling" the data values with the sampling rate etc...
    imsave('sorted/audmages/'+ genre +'/'+ audFileName +'.png', newdata)
    
    #print 'Finished round of doAudmage!\n'
    #sys.exit()
  #End doAudmage loop
  return True
#End doAudmage function


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
  if Spect and os.path.isdir('sorted/spect'):
    item = 'spect'
  elif Audmage and os.path.isdir('sorted/audmage'):
    item = 'audmage'
  else:
    print 'No image set was chosen'
    sys.exit()

  genres = os.listdir('sorted/'+ item) #list of genres
  gCount = [] #total images for each genre
  iTotal = 0 #total number of images among all genres
  for genre in genres:
    tmpCount = len(os.listdir('sorted/'+ item +'/'+ genre)) #count of files listed within genre
    gCount.append(tmpCount) #save count for this genre
    iTotal += tmpCount #add this genre count to total
  
  #At this point "item" value matters.. both or one?
  i = 0
  while i < len(genres):
    tracks = os.listdir('sorted/'+ item +'/'+ str(genres[i]))
    shuffle(tracks) #mix up the tracks to scramble datasets each time
    
    #split up the tracks in this genre for each dataset item (train/test/validate)
    p1tracks = tracks[:int(gCount[i]*p1)] #only take p1 percent of these tracks
    p2tracks = tracks[int(gCount[i]*p1):int((gCount[i]*p1)+(gCount[i]*p2))] #take p2 percent of these tracks
    p3tracks = tracks[int((gCount[i]*p1)+(gCount[i]*p2)):] #all the rest... p3 percent of these tracks..

    #copy the files to the dataset directories
    for track in p1tracks:
      copyfile('sorted/'+ item +'/'+ genres[i] +'/'+ track, 'dataset/'+ item +'/test/'+ track)
      if item == 'spect' and Audmage:
        copyfile('sorted/audmage/'+ genres[i] +'/'+ track, 'dataset/audmage/test/'+ track)
      if item == 'audmage' and Spect:
        copyfile('sorted/spect/'+ genres[i] +'/'+ track, 'dataset/spect/test/'+ track)
  
    for track in p2tracks:
      copyfile('sorted/'+ item +'/'+ genres[i] +'/'+ track, 'dataset/'+ item +'/train/'+ track)
      if item == 'spect' and Audmage:
        copyfile('sorted/audmage/'+ genres[i] +'/'+ track, 'dataset/audmage/train/'+ track)
      if item == 'audmage' and Spect:
        copyfile('sorted/spect/'+ genres[i] +'/'+ track, 'dataset/spect/train/'+ track)

    for track in p3tracks:
      copyfile('sorted/'+ item +'/'+ genres[i] +'/'+ track, 'dataset/'+ item +'/validate/'+ track)
      if item == 'spect' and Audmage:
        copyfile('sorted/audmage/'+ genres[i] +'/'+ track, 'dataset/audmage/validate/'+ track)
      if item == 'audmage' and Spect:
        copyfile('sorted/spect/'+ genres[i] +'/'+ track, 'dataset/spect/validate/'+ track)

    i += 1 #increment genre index
  #End While Loop
  
  return True
#End generateSet function

###############################
# Check for command arguments #
###############################
print "\n### Audmage - Audio collection genre sorter ###\n"

#option flags default values
Audio = False #Flag to sort the audio files by genre
Spect = False #Flag to create and sort spectrograms by genre
Audmage = False #Flag to create and sort audmages by genre
DataF = False #Flag to create a dataset from sorted images
ReTag = False #Flag to modify incorrect genre tags in audiofiles
              #(Based on tracks.csv metadata file data)

#Array of audio file paths and genres
Fg = []

#open and read the tracks csv file
try:
  ifile = open("fma_metadata/tracks.csv")
  meta = csv.reader(ifile)
except IOError:
  print 'Unable to open '+ metapath + '/tracks.csv'
  sys.exit()

#Do we have any commandline arguments?
if len(sys.argv) > 1:

  #1 - the path to the audio collection
  if not os.path.isdir(sys.argv[1]) == True:
    print "Error, " + str(sys.argv[1]) +" is not a directory or doesn't exist\n"
    sys.exit()

  #2 - True/False (Should a dataset be generated after processing?)
  if len(sys.argv) > 2:
    if sys.argv[2] == "True":
      DataF = True
  
  #3+ - check for options
  if len(sys.argv) > 3:
    #split up command arguments taking all arguments after aug 1
    #and loop through each option to check which it is.
    for option in sys.argv[3:]:
      if option.lower() == 'audio':
        #User wants to sort the audio files
        Audio = True
      elif option.lower() == 'spect' or option == 'spectrogram':
        #User wants to create spectrogram images
        Spect = True
      elif option.lower() == 'audmage':
        #User wants to create audmages
        Audmage = False
      elif option.lower() == 'retag':
        #User wants to retag the audio files
        ReTag = True
    #end option loop
  else:
    #No extra options found, default is all options
    Audio = Spect = ReTag = True #Audmage = false for now
  
  #Finished setting up options
  #4 - Let's begin
  if Audio:
    print 'Attempting to sort audio files into genres...'
  else:
    print 'Gathering Audio file locations and genres...'

  #Begin the audio file search/sort
  if audSearch(sys.argv[1]):
    if Audio:
      print 'Audio collection was sorted successfully!'
    else:
      print 'Successfully gathered audio file locations and genres!'
    
    #Convert to spectrogram?
    if Spect == True:
      doSpect(Fg)

    #Convert to audmage?
    #if Audmage == True:
    #  doAudmage(Fg)

    #5 - Generate a dataset?
    if DataF:
      #Can't create a dataset unless there are images to use..
      if os.path.isdir('sorted/spect') or os.path.isdir('sorted/audmage'):
        print 'Generating dataset from sorted collection...\n'
        if generateSet(.8, .1, .1):
          print 'Dataset has been generated successfully!\n'
      else:
        print 'Could not generate Dataset, no images were created!\n'
else:
  #No command line arguments provided, need data collection path!
  print "Error, no path to audio files provided! \nExiting...\n"
  sys.exit()
