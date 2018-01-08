from os import listdir
from os.path import isfile, join
import cv2

def emotions10k(start_index, end_index):
	onlyfiles = sorted([f for f in listdir('10kFaceImages') if isfile(join('10kFaceImages', f))])
	emotions = []
	f = open('emotions_10k_' + str(start_index) + '_' + str(end_index) + '.txt', 'a+')
	# for image in onlyfiles:
	flag = 'ok'
	for i in xrange(start_index, end_index + 1):
		try:
			img = cv2.imread('10kFaceImages/' + onlyfiles[i])
			cv2.imshow('Image', img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			emotions.append(raw_input("Image no. " + str(i) + ": "))
		except Exception as e:
			f.write("\n".join(emotions))	
			flag = 'error'
	if flag == 'ok':
		f.write("\n".join(emotions))
	f.close()

def emotionsJaffe():
	onlyfiles = sorted([f for f in listdir('jaffe') if isfile(join('jaffe', f)) and 'tiff' in f])
	emotions = []
	for image in onlyfiles:
		img = cv2.imread('jaffe/' + image)
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		emotions.append(int(raw_input()))
	f = open('emotions_jaffe.txt', 'w+')
	f.write("\n".join(emotions))
	f.close()

# emotionsJaffe()
"""
Call one of the below, comment the rest. 
Total no. of images = 10168
To ease the process, change the transparency level of your command line so that you can see both the command line and image at the same time.
In Windows, do the following: 
* 	Right click on the top left corner icon of your command window.
* 	Go to Properties.
* 	Select the Colors tab.
*	Change the transparency using the slider on the bottom.

Now keep typing the cmd and image will change along with that.
"""
# emotions10k(0, 3388) 		#		Arpit
# emotions10k(3389, 6778) 	#		Lakshay
# emotions10k(6778, 10167)	#		Suruchi