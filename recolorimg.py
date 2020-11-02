import numpy as np
import argparse
import cv2
import os

image_arg="m.jpg" #path to input black and white image
prototxt_arg="model\colorization_deploy_v2.prototxt" #path to Caffe prototxt file
model_arg="model\colorization_release_v2.caffemodel" #path to Caffe pre-trained model
points_arg="model\pts_in_hull.npy" #path to cluster center points

# load our serialized black and white colorizer model and cluster center points from disk
print("LOADING MODEL")
net = cv2.dnn.readNetFromCaffe(prototxt_arg, model_arg)
pts = np.load(points_arg)

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# load image and scale the pixel intensities to range [0,1], then convert from BGR to Lab color space
image = cv2.imread(image_arg)
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# passing L channel through the network to predict the ab channel
'print("COLORIZING IMAGE")'
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# resize the predicted 'ab' volume to the same dimensions as our input image
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# concatenate the L channel (input image) with the ab channel
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# convert colorized image from Lab to RGB
# clip any values that fall outside the range [0, 1]
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

#convert current floating point representation of colorized image to an unsigned 8-bit int in range [0,255]
colorized = (255 * colorized).astype("uint8")

# save the colorized image
filename="colorized_"+image_arg
cv2.imwrite(filename, colorized)

# show the original and output colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)