# A Tool for Manually Sample Road Center Points from Remote Sensing Images
This tool is written in python3.5, Please install python3.5 or above before running, as well as numpy and PIL
## Usage
### Run the program
Run main.py, a GUI interface is shown as below
<img alt='Main interface' src='images/main.jpg?raw=true' />
### Load an image
Click File->Load an Image, Select an image which you want to sample points from. The tool is displays the image and shows the grid over the image. The default size of grid is 64 x 64. 
<img alt='Load image' src='images/load.jpg?raw=true' />
### Change the grid size
The size of the grid determins the size of the sampled patch. If the patch size is different to default size, You can change it. Click View->Grid->Edit, an dialog will pop up. You can input parameters of grid in it, as shown below. 
<img alt='Chang grid' src='images/gridsetting.jpg?raw=true' />
### Drag
If the size of the picture is too large, you can drag the picture in order to easily view the different parts. Right Click, select "Dray" command and hold the left button of the mouse and drag the image.
### Scaling
If you want to see the details of the image, you can zoom in and out of the image. Just scroll the mouse wheel.
### Select mark command
Right click the image. Select the "Mark Foreground by Points" command in the popup menu if you want to mark road center points. Select the "Mark Background by Points" command in the popup menu if you want to sample a patch that there is no road in it (combining such type of samples, The trained the CNN model has the ability to determine the existence of road in a patch).
<img alt='popup menu' src='images/popmenu.jpg?raw=true' />
### Mark road center points
When you select "Mark Foreground by Points", then left click the any patch shown in the grid. We stipulate that this point should be approximately at the center of the road measured by human eyes. The road center points will mark by red disk. Note that keep the grid on during the sampling process
<img alt='mark road center' src='images/roadcenter.jpg?raw=true' />
### Mark background samples
When you select "Mark Background by Points", then left click any patch show in the grid.We specify that these points can be placed anywhere in the patches which do not have roads, The points will mark by yellow disk
<img alt='mark background' src='images/background.jpg?raw=true' />
### Points fine-tuning
When you find that some positions are not placed accurately enough, you can fine-tune the points. Operation: move the mouse to the point you want to adjust, the mouse will turn into a hand cursor, and then hold down the left button and drag to a more accurate position
###Delete points
When you find that some points are redundant, you can delete them. Operating:move the mouse to the point you want to adjust, the mouse will turn into a hand cursor, and then hold down the left button and drag outside the image.
###Generate the sampled patches
So far, you have just marked on the image which patches are positive samples (the patches with the center of the road) and which are negative samples (the background patches), but the program has not generated these samples. Click File->gen point samples, then you can select a folder to store these trainging samples. We will generate three files for each patch, which are an image slice, a point truth mask image, and a point truth data file. There are samples shown as below. The Components of file name are image file name, ,road existence, top-left corner coordinate. For example: 10078660_15_r0_0960_0448_img.png, 10078660_15 is the name of the image file from which the patch is cropped, r0 indicates that there is no road in the patch, 0960_0448 indicates that the coordinates of the top-left corner of the patch in the image are (960, 448)

<img alt='samples' src='images/samples.jpg?raw=true' />

