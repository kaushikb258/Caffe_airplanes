import numpy as np
import matplotlib.pyplot as plt


filename = "/home/kb/airplanes_caffe/AlexNet/alexnet.log"
log = open(filename, "r").readlines()

it = []
a1 = []
a5 = []


for line in log:
 if "Testing net (#0)" in line:	
  if len(line.split()) == 9:
   it.append(line.split()[5].split(',')[0])
 if "accuracy_at_1" in line:
  if len(line.split()) == 11:
   a1.append(line.split()[-1]) 
 if "accuracy_at_5" in line:
  if len(line.split()) == 11:
   a5.append(line.split()[-1])


it = np.array(it).astype(np.int)
a1 = np.array(a1).astype(np.float)
a5 = np.array(a5).astype(np.float)

print "shapes ", it.shape, a1.shape, a5.shape

plt.plot(it,a1,label="top 1")
plt.plot(it,a5,label="top 5")
plt.xlabel("Iteration #",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)  
plt.legend(fontsize=15)
#plt.show()
plt.savefig("/home/kb/airplanes_caffe/AlexNet/learning_curve.png")
