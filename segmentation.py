import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from imAdjust import imAdjust
from cellpose import io, models, utils
from cellpose.plot import image_to_rgb,mask_overlay
from convert2Binary import calculateCentroids
from PIL import Image
from skimage import measure

def checkSegmentation(outDir,image,crop,channel,adjustment,modelType,diameter='None'):

    # Adjust image for contrast and brightness
    imAdjusted = adjustCrop(outDir,image,crop,channel,adjustment)

    # Create cellpose model
    model = models.Cellpose(gpu=False,model_type=modelType)

    # Run cellpose
    chan = [0,0]
    masks,flows = model.eval(imAdjusted,diameter=diameter,channels=chan)[:2]
    
    print("Using default cellprob_threshold=0.0,low_threshold=0.4.")

    # Show cellpose output
    plotResults(imAdjusted,masks,flows[0],chan)
    print("Segmented",str(len(np.unique(masks)[1:])),"cells.")

    return imAdjusted,model,flows

def tweakThresh(segmentation,cellprob_threshold=0.0,flow_threshold=0.4):

    # Load segmentation
    imAdjusted,model,flows = segmentation

    # Recompute masks based on new thresholds
    masks = model.cp._compute_masks(flows[1],flows[2],cellprob_threshold=cellprob_threshold,flow_threshold=flow_threshold,resize=imAdjusted.shape[-2:])[0]

    # Show cellpose output
    chan = [0.0]
    plotResults(imAdjusted,masks,flows[0],chan)
    print("Segmented",str(len(np.unique(masks)[1:])),"cells.")

def runSegmentation(outDir,image,channel,modelType,diameter=None,cellprob_threshold=0.0,flow_threshold=0.4,minArea=5) :

    t = time.time()

    cropDir,crops = getCrops(outDir,image)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False,model_type=modelType)

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    chan = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    # channels = [[2,3], [0,0], [0,0]]

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended) 
    # diameter can be a list or a single number for all images

    # you can run all in a list e.g.
    # >>> imgs = [io.imread(filename) in for filename in files]
    # >>> masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
    # >>> io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
    # >>> io.save_to_png(imgs, masks, flows, files)

    # or in a loop

    counts = list()

    raw = cropDir.replace('Cropped_Plate_' + str(image) + '_','',1)
    segmentationDir = outDir + 'Image_Segmentation/Segmentation_Plate_' + str(image) + '_Channel_' + str(channel) + '_' + raw.split('.')[0]

    # Create output directories if necessary
    try :
        os.makedirs(segmentationDir)
        os.mkdir(outDir + 'Image_Counts')
    except :
        pass

    i = 1
    for filename in crops:

        print('Segmenting',str(i) + '/' + str(len(crops)),'crops.')

        # Run cellpose on cropped image
        cropid = int(filename.split('_')[4])
        im = io.imread(outDir + 'Cropped_Images/' + cropDir + '/' + filename)[channel-1,:,:]
        masks,flows,styles,diams = model.eval(im,diameter=diameter,channels=chan,cellprob_threshold=cellprob_threshold,flow_threshold=flow_threshold)
        
        # Calculate outlines and centroids
        outlines = utils.masks_to_outlines(masks)
        outlinesLabeled = measure.label(np.where(outlines > 0,0,255),background=0,connectivity=1)
        bgLabeled = outlinesLabeled[0,0]
        outlinesLabeled[outlinesLabeled==bgLabeled] = 0
        centroids = calculateCentroids(outlines,outlinesLabeled,minArea)

        segmentation = [cropid,masks,flows,styles,diams,outlines,centroids]
        counts.append([cropid,len(np.unique(masks)[1:])])

        # Save segmentation as .npy
        np.save(segmentationDir + '/Segmentation_Plate_' + str(image) + '_Channel_' + str(channel) + '_Crop_' + str(cropid) + '_' + raw.split('.')[0],np.array(segmentation,dtype=object))

        i += 1

    # Sort counts by crop
    counts = sorted(counts,key=lambda x:x[0])

    # Save crop counts as .txt
    with open(outDir + 'Image_Counts/Counts_Plate_' + str(image) + '_Channel_' + str(channel) + '_' + raw.split('.')[0] + '.txt', 'w') as f:
        json.dump(counts, f)

    # Print execution time
    print('Finished. Took',str(int(time.time()-t)),'seconds.')

def seeSegmentation(outDir,image,crop,channel,adjustment) :

    # Adjust image for contrast and brightness
    imAdjusted = adjustCrop(outDir,image,crop,channel,adjustment)

    # Get paths to segmentation directory and the desired crop
    images = [outDir + 'Image_Segmentation/' + i for i in os.listdir(outDir + 'Image_Segmentation/') if os.path.isdir(os.path.join(outDir + 'Image_Segmentation/',i)) and i.startswith('Segmentation_Plate_' + str(image) + '_Channel_' + str(channel) + '_')]
    crops = [images[0] + '/' + i for i in os.listdir(images[0]) if os.path.isfile(os.path.join(images[0],i)) and i.startswith('Segmentation_Plate_' + str(image) + '_Channel_' + str(channel) + '_Crop_' + str(crop) + '_')]
    
    # Load segmentation data
    segmentation = np.load(crops[0],allow_pickle=True)
    masks = segmentation[1]
    outlines = segmentation[5]
    centroids = segmentation[6]

    # Show cellpose output
    masks = np.where(masks > 0, True, False)
    masks = np.array(Image.fromarray(masks).convert('RGB'))
    outX, outY = np.nonzero(outlines)
    masks[outX, outY] = np.array([255,75,75])
    
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(imAdjusted,vmin=0,vmax=255,cmap='gray')
    ax1.set_title('original image')
    ax1.axis('off')

    ax2 = fig.add_subplot(1,2,2,sharex=ax1,sharey=ax1)
    ax2.imshow(masks)
    y,x = zip(*centroids)
    ax2.scatter(x,y,s=3)
    ax2.set_title('masks')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    print("Segmented",str(len(centroids)),"cells.")

def getCrops(outDir,image) :

    # Get paths to crops
    imageDir = outDir + 'Cropped_Images/'
    images = [i for i in os.listdir(imageDir) if os.path.isdir(os.path.join(imageDir,i)) and i.startswith('Cropped_Plate_' + str(image) + '_')]
    crops = [i for i in os.listdir(imageDir + images[0]) if os.path.isfile(os.path.join(imageDir + images[0],i)) and i.startswith('Cropped_Plate_' + str(image) + '_Crop_')]
    
    return images[0], crops

def adjustCrop(outDir,image,crop,channel,adjustment) :

    cropDir, crops = getCrops(outDir,image)
    crops = [outDir + 'Cropped_Images/' + cropDir + '/' + i for i in crops if i.startswith('Cropped_Plate_' + str(image) + '_Crop_' + str(crop) + '_')]

    imCropped = io.imread(crops[0])[channel-1,:,:]
    imAdjusted = imAdjust(imCropped,*adjustment)

    return imAdjusted

def plotResults(imAdjusted,masks,flows,chan):

    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(1,3,1)
    img0 = imAdjusted.copy()
    if img0.shape[0] < 4:
        img0 = np.transpose(img0,(1,2,0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0,channels=chan)
    else:
        if img0.max()<=50.0:
            img0 = np.uint8(np.clip(img0*255, 0, 1))
    ax1.imshow(imAdjusted,vmin=0,vmax=255,cmap='gray')
    ax1.set_title('original image')
    ax1.axis('off')

    outlines = utils.masks_to_outlines(masks)
    overlay = mask_overlay(img0, masks)

    ax2 = fig.add_subplot(1,3,2,sharex=ax1,sharey=ax1)
    outX, outY = np.nonzero(outlines)
    imgout= img0.copy()
    imgout[outX, outY] = np.array([255,75,75])
    ax2.imshow(imgout)
    ax2.set_title('predicted outlines')
    ax2.axis('off')

    ax3 = fig.add_subplot(1,3,3,sharex=ax1,sharey=ax1)
    ax3.imshow(overlay)
    ax3.set_title('predicted masks')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()