import cv2 as cv
import json
import numpy as np
import os
import tifffile
import time
from imAdjust import imAdjust
from imshowFast import imshow
from matplotlib import pyplot as plt
from PIL import Image
from skimage import measure

def checkBinary(raw,channel,adjustment,performance='normal',binaryMode='std',minArea=50,**kwargs):

    stdThresh = kwargs.get('stdThresh',5)
    threshold = kwargs.get('threshold',4000)

    # Presets for performance modes
    previewScale,strideScale = {'normal':(0.5,0.25),'fast':(0.25,0.5),'fancy':(1,0.25)}[performance]

    # Read specified channel of image into a NumPy array
    im = tifffile.imread(raw)[channel-1,:,:]

    # Scale and adjust image for contrast and brightness as necessary
    if previewScale != 1 :
        imPreview = cv.resize(im,dsize = tuple([int(previewScale * i) for i in im.shape]))
        imAdjusted = imAdjust(imPreview,*adjustment)
    else :
        imAdjusted = imAdjust(im,*adjustment)

    # Binarize image and obtain coordinates of region centroids
    imBinary,centroids = binarize(im,binaryMode,stdThresh,threshold,minArea)

    # Set up plots for raw and binarized images
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
    imshow(strideScale,ax1,imAdjusted,vmin=0,vmax=255,cmap='gray',extent=[0,im.shape[1],im.shape[0],0])
    imshow(strideScale,ax2,imBinary,vmin=0,vmax=1,cmap='gray',interpolation='nearest')
    
    # Add scatter plot for centroids
    y,x = zip(*centroids)
    ax2.scatter(x,y,s=3)

    # Show plot
    plt.show()

def convert2Binary(rawDir,outDir,channel,binaryMode='std',minArea=50,**kwargs):
    
    t = time.time()
    stdThresh = kwargs.get('stdThresh',5)
    threshold = kwargs.get('threshold',4000)

    # Iterate through raw images and save binarized images with region centroid lists
    i = 1
    for raw in os.listdir(rawDir):
        try:
            # Read specified channel of image into a NumPy array
            im = tifffile.imread(rawDir + raw)[channel-1,:,:]

            # Binarize image and obtain coordinates of region centroids
            imBinary,centroids = binarize(im,binaryMode,stdThresh,threshold,minArea)

            # Create output directories if necessary
            try :
                os.mkdir(outDir + 'Binary_Images')
                os.mkdir(outDir + 'Coordinate_Lists')
            except :
                pass

            # Convert binarized image array to Pillow Image
            imBinary = Image.fromarray(imBinary)

            # Save Pillow Image as compressed .tif
            imBinary.save(outDir + 'Binary_Images/Binary_Image_' + str(i) + '_Channel_' + str(channel) + '_' + raw,compression='tiff_lzw')

            # Save region centroid list as .txt
            with open(outDir + 'Coordinate_Lists/Coordinate_List_Image_' + str(i) + '_Channel_' + str(channel) + '_' + raw.split('.')[0] + '.txt','w') as f:
                json.dump(centroids,f)

            i += 1
        except:
            continue

    # Print execution time
    print("Binarized",str(i-1),"images. Took",str(int(time.time()-t)),"seconds.")

def binarize(im,binaryMode,stdThresh,threshold,minArea):

    # Binarize raw image using either otsu algorithm, standard deviation, or manual thresholding
    if binaryMode == 'otsu':
        threshold,imBinary = cv.threshold(im,128,255,cv.THRESH_OTSU)
        imBinary = imBinary > 0
        print("threshold =",threshold)
    elif binaryMode == 'std':
        threshold = np.mean(im) + stdThresh * np.std(im) # Mean intensity + z-score * standard deviation
        print("threshold =",threshold)
        imBinary = np.where(im > threshold,True,False)
    elif binaryMode == 'manual':
        imBinary = np.where(im > threshold,True,False)
    else:
        print("Invalid binary mode.")

    # Label regions on binarized image and obtain centroids
    imLabeled = measure.label(imBinary,connectivity=2)
    centroids = calculateCentroids(imBinary,imLabeled,minArea)

    return imBinary,centroids

def calculateCentroids(im,imLabeled,minArea) :

    # Get properties of each labeled region
    regionProps = measure.regionprops(imLabeled,cache=False)

    # Mark small objects based on area threshold
    smallObjects = list()
    i = 0
    for region in regionProps:
        if region.area < minArea:
            for coords in region.coords:
                im[coords[0],coords[1]] = False
            smallObjects.append(i)
        i += 1

    # Filter out regions marked as small objects
    regionProps = [i for j, i in enumerate(regionProps) if j not in smallObjects]

    # Obtain centroids of filtered regions
    centroids = [p.centroid for p in regionProps]

    return centroids