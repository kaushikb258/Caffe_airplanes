import glob
import os
import random
import shutil
import numpy as np


TRAIN_SET = []
VAL_SET = []

caldir = "/home/kb/airplanes_caffe/data"


types = []


#Fine number of classes
filename = caldir + "/fgvc-aircraft-2013b/data/images_family_train.txt"
log = open(filename,"r").readlines()
for line in log:
 ac = ''.join(line.split()[1:])
 if ac not in types:
  types.append(ac)

filename = caldir + "/fgvc-aircraft-2013b/data/images_family_val.txt"
log = open(filename,"r").readlines()
for line in log:
 ac = ''.join(line.split()[1:])
 if ac not in types:
  types.append(ac)

print "number of aircraft types = ", len(types)


# Read training set
filename = caldir + "/fgvc-aircraft-2013b/data/images_family_train.txt"
log = open(filename,"r").readlines()
for line in log: 
 i = line.split()[0] 
 img = caldir + "/fgvc-aircraft-2013b/data/images/" + str(i) + ".jpg"
 ac = ''.join(line.split()[1:]) 
 typ = str(types.index(ac)) 
 TRAIN_SET.append((img,typ))


# Read validation set
filename = caldir + "/fgvc-aircraft-2013b/data/images_family_val.txt"
log = open(filename,"r").readlines()
for line in log: 
 i = line.split()[0] 
 img = caldir + "/fgvc-aircraft-2013b/data/images/" + str(i) + ".jpg"
 ac = ''.join(line.split()[1:]) 
 typ = str(types.index(ac)) 
 VAL_SET.append((img,typ))


np.random.shuffle(TRAIN_SET)
np.random.shuffle(VAL_SET)



#Write the distribution into separate text files
try:
  os.mkdir(caldir + "/lmdb")
except:
  pass

f = open(caldir + "/lmdb/train.txt", "w")
for _entry in TRAIN_SET:
 f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()

f = open(caldir + "/lmdb/val.txt", "w")
for _entry in VAL_SET:
 f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()


