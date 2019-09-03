# USAGE : python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb


# IMPORTING NECESSARY PACKAGES


import imutils
from imutils import contours
from imutils.object_detection import non_max_suppression
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import cv2
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import math
from scipy import ndimage
import bisect

# CONSTRUCTING ARGUMENT PARSER AND ADDING ARGUMENTS

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
        help="path to input image")
ap.add_argument("-east", "--east", type=str,
        help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
        help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
        help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
        help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())


# LOADING IMAGE AND GRABBING ITS DIMENSIONS


image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]


# SETTING NEW WIDTH AND HEIGHT AND DETERMINING THE RATIO OF CHANGE IN THEM


(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)


# RESIZING AND GRABBING DIMENSIONS OF THE IMAGE


image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]


# DEFINING 2 REQUIRED O/P LAYERS THAT ARE OF PROBABILITIES AND CO-ORDINATES


layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]


# LOADING PRE-TRAINED EAST MODEL


print("\n-> LOADING TEXT DETECTOR ...")
net = cv2.dnn.readNet(args["east"])


# CONSTRUCTING BLOB FROM IMAGE AND PASSING IT TO MODEL AND RETRIEVING SCORES AND GEOMETRIES


blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()


# SHOW TIMING INFO


print("-> TEXT DETECTION TOOK : {:.6f} seconds".format(end - start))


# GRABBING SCORES VOLUME AND INITIALIZING RECTS AND CONFIDENCES


(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []


# FINDING AND SELECTING THE POTENTIAL BOUNDUNG BOXES


for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < args["min_confidence"]:
                        continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])


# APPLYING NON-MAX SUPPRESSION


boxes = non_max_suppression(np.array(rects), probs=confidences)
img_col = []

fimg = []
# LOOPING OVER BOUNDING BOXES


for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 0), 1)
        img_col.append((startX, startY, endX, endY))         

        

# FINDING MAX WIDTH AND HEIGHT OF THE BOXES


maax = 0
maay = 0
cnt = 0
for (startX, startY, endX, endY) in img_col:
        if maax < abs(startX - endX):
                maax = abs(startX - endX)
        if maay < abs(startY - endY):
                maay = abs(startY - endY)        

maax = int(maax*2)


# STITCHING EACH BOXES INTO ONE IMAGE


for (startX, startY, endX, endY) in img_col:
        hori = orig[startY:endY, startX:endX]
        hori = cv2.resize(hori,(maax,maay))
        fimg.append(hori)
        break

for (startX, startY, endX, endY) in img_col:

        crp_image = orig[startY:endY, startX:endX]
        crp_image = cv2.resize(crp_image,(maax,maay))
        if cnt != 0:
                hori = np.hstack((hori,crp_image))
        cnt = cnt + 1        
        fimg.append(crp_image)

# CONVERT TO BINARY


image = cv2.resize(hori,(hori.shape[1],30))
gimage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
(thresh,mask) = cv2.threshold(gimage,127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Text Detection",orig)
cv2.imshow("Binary Image",mask)



with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	model = load_model('model.h5')



print('Model successfully loaded')
# SEGMENTING EACH CHARACTERS
text = ""
def form(img):
    def getBestShift(img):
        cy,cx = ndimage.measurements.center_of_mass(img)
        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty
    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
    gray=img
    gray = cv2.resize(255-gray, (40, 40))
    ret, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    while np.sum(gray[0]) == 0:
            gray = gray[1:]
    while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)
    while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
    while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)
    rows,cols = gray.shape
    if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
    else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    flatten = gray.flatten() / 255.0
    return gray

	
	
def giveChar(img,model):
	label = np.argmax(model.predict(img.reshape(1,28,28,1)))
	mapp = {36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',45:'r'}
	if label<10:
		return str(label)
	elif label<36:
		return str(chr(ord('A')+label-10))
	else:
		return str(mapp[label])
	print(label)
	return label
	
def giveChar2(img,model):
	label = np.argmax(model.predict(img.reshape(1,28,28,1)))
	if label<10:
		return str(label)
	elif label<36:
		return str(chr(ord('A')+label-10))
	else:
		return str(chr(ord('a')+label-10))
	print(label)
	return label
	
	
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
index = 1
width = 28
height = 28
dim = (width, height)

X=[]
for contour in contours:
	[x, y, w, h] = cv2.boundingRect(contour)
	bisect.insort(X,x)
print(X)
textdict={}

for contour in contours:
	[x, y, w, h] = cv2.boundingRect(contour)
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),1)
	cropped = mask[y :y +  h , x : x + w]

	cropped = form(cropped)
	plt.subplot(len(contours),1,index)
	plt.imshow(cropped)
	if X.index(x) in textdict:
		textdict[X.index(x)+1]=giveChar(cropped,model)
	else:
		textdict[X.index(x)]=giveChar(cropped,model)
	s =str(X.index(x)) + '.png'
	if w < 5 and h < 5:
	   continue
	constant= cv2.copyMakeBorder(cropped,5,5,5,5,cv2.BORDER_CONSTANT,(0,0,0))
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.dilate(constant,kernel,iterations = 1)
	erosion = cv2.erode(erosion,kernel,iterations = 1)
	erosion = cv2.dilate(erosion,kernel,iterations = 1)
	resized = cv2.resize(erosion, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite(s , resized)
	index = index + 1

for i in range(len(X)):
	text += str(textdict[i])
	
print("Detected text: "+text)
plt.show()
cv2.imshow("Detected", image)
cv2.waitKey(0)
