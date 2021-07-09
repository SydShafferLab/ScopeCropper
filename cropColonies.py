import json
import os
import tifffile
import time
from imshowFast import imshow
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['keymap.home'].remove('r')

def cropRaw(rawDir,outDir,image,channel):

    # Define paths based on ScopeCropper folder structure
    binaryDir = outDir + 'Binary_Images/'
    coordListsDir = outDir + 'Coordinate_Lists/'

    # Get paths to binary image and its cooresponding coordinate list
    images = [i for i in os.listdir(binaryDir) if os.path.isfile(os.path.join(binaryDir,i)) and i.startswith('Binary_Image_' + str(image) + '_Channel_' + str(channel) + '_')]
    coordLists = [i for i in os.listdir(coordListsDir) if os.path.isfile(os.path.join(coordListsDir,i)) and i.startswith('Coordinate_List_Image_' + str(image) + '_Channel_' + str(channel) + '_')]

    # Read binary file 
    imBinary,coords = readBinary(binaryDir + images[0],coordListsDir + coordLists[0])
    y,x = zip(*coords)

    cropList = list()
    scatterList = list()
    actions = list()

    def onPress(event):

        if event.key in ('w','e','r') :
            bounds = ax2.viewLim.bounds
            if event.key == 'w':
                actions.append('w')

                # Label colony region with lime and add to crop list
                cropList.append(Rectangle((0,0),0,0,facecolor='none',edgecolor='lime',linewidth=1))
                label(cropList[-1],'lime',bounds)
            elif event.key == 'e':
                actions.append('e')

                # Label noise region with red
                badRect = Rectangle([0,0],0,0,facecolor='none',edgecolor='r',linewidth=1)
                label(badRect,'r',bounds)
            elif event.key == 'r':

                # Remove last label
                ax1.patches[-1].remove()
                scatterList[-1][0].remove()
                scatterList[-1][1].remove()
                scatterList.pop()

                # Remove region from crop list if necessary
                if actions[-1] == 'w':
                    cropList.pop()

                # Remove last action record
                actions.pop()

            # Reset axes viewing limits
            ax1.set_xlim(0,imBinary.shape[1])
            ax1.set_ylim(imBinary.shape[0],0)
            ax2.set_xlim(0,imBinary.shape[1])
            ax2.set_ylim(imBinary.shape[0],0)

            # Update plots
            fig.canvas.draw()
        elif event.key == 't':
            t = time.time()

            # Get bounding box for crops
            crops = [Rectangle.get_bbox(r).get_points() for r in cropList]

            # Read raw image into a NumPy array
            raw = images[0].replace('Binary_Image_' + str(image) + '_Channel_' + str(channel) + '_','',1)
            print("Opening",raw,".")
            im = tifffile.imread(rawDir + raw)
            
            # Create output directory if necessary
            croppedDir = outDir + 'Cropped_Images/Cropped_Plate_' + str(image) + '_' + raw.split('.')[0]
            try :
                os.makedirs(croppedDir)
            except :
                pass
            
            # Crop raw image and save as compressed .tif
            i = 1
            for crop in crops :
                imCropped = im[:,int(crop[1][1]):int(crop[0][1]),int(crop[0][0]):int(crop[1][0])]
                print("Saving",str(i) + '/' + str(len(crops)),"crops.")
                tifffile.imsave(croppedDir + '/' + 'Cropped_Plate_' + str(image) + '_Crop_' + str(i) + '_' + raw,imCropped,photometric='minisblack',compression=8)
                i += 1

            # Print execution time
            print("Finished. Took",str(int(time.time()-t)),"seconds.")

    # Set up GUI
    fig = plt.figure(figsize=(12,6))
    fig.canvas.mpl_connect('key_press_event',onPress)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    imshow(0.25,ax1,imBinary,vmin=0,vmax=1,cmap='gray',interpolation='nearest')
    imshow(0.25,ax2,imBinary,vmin=0,vmax=1,cmap='gray',interpolation='nearest')
    ax1.scatter(x,y,s=3)
    ax2.scatter(x,y,s=3)
    ax1.set_title('Crops')
    ax2.set_title('Selection')
    ax2.set_xlabel('w = mark colony, e = mark noise, r = undo, t = save')
    plt.tight_layout()
    plt.show()

    def label(rect,scatterColor,bounds):

        # Add region rectangle and overlay labeled scatter
        rect.set_bounds(*bounds)
        ax1.add_patch(rect)
        newCoords = cropCoords(coords,bounds)
        ny,nx = zip(*newCoords)
        scatterList.append((ax1.scatter(nx,ny,s=3,color=scatterColor),ax2.scatter(nx,ny,s=3,color=scatterColor)))

    def cropCoords(coords,bounds):

        # Get list of coordinates within crop region
        coordsCropped = [coord for coord in coords if (bounds[0] < coord[1] < bounds[0] + bounds[2]) & (bounds[1] + bounds[3] < coord[0] < bounds[1])]

        return coordsCropped

def readBinary(binary,coordList):

    # Read binary image into a NumPy array
    imBinary = tifffile.imread(binary)

    # Read region coordinate list
    with open(coordList,'r') as f:
        coords = json.load(f)

    return imBinary,coords