# DeepWindow
A Sliding Window based on Deep Learning for Road Extraction in Remote Sensing
<p>
  <b>Abstract: </b>
The template matching methods are commonly applied to extract the road network in remote sensing images. However, the matching rules are manually specified in the traditional template matching methods which may make the extracted road center line deviated from the road center in complex cases and even result in overall tracking errors. The methods based on semantic segmentation exploiting deep learning network greatly increases the extraction performance, nevertheless, the annotation of the training samples is expensive and the post-process is error-prone. We propose DeepWindow, a new method to automatically extract the road network from remote sensing images. DeepWindow uses an iterative sliding window guided by a CNN-based decision function to track the road network directly from the images without the guidance of road segmentation. In our method, the tracking seeds are automatically searched, which get rid of the artificial intervention and makes the process of road tracking process fully automated combing with the road direction estimation. Furthermore, the point annotations for CNN training greatly reduces the cost of model training. Extensive and comprehensive experiments performed on the extraction of roads and retinal vessels indicate that our method is effective and can be applied for tracking topology in other scenes.
  </p>

# Dataset
We use <a href='http://www.cs.toronto.edu/~vmnih/data/'>Massachusetts Roads Dataset</a>. 

# Training Samples
To train the road center estimation model, we cut fifty patches with the resolution of 64ⅹ64 from each image in the training set, and finally obtain 55,400 training samples. There are some of our training samples. where the ground truths are gaussians center at the road center points shown in cyan. Note that there are some samples without ground truth for no roads inside these patches.
<img src='resources/trainingsamples.png' width='100%'/>

# Predicting Samples
Exhibition of more predicted samples, in where the predicted road center points are presented in bold red diamonds. Note that there are some samples without road center points due to none roads in these patches.
<img src='resources/predictsamples.png' width='100%'/>

# Videos
Our algorithm works in a sliding window mode. It is a evolving process step by step. We release some videos to show the trackiing process.

<video src='resources/video2.mp4' />
Video 1. A tracking process on a small image cropped from an oringin test image.
<video src='resources/video1.mp4' />
Video 2. A tracking process on the oringin test image.

# Output Visualization
Exhibition of some road tracking results on the Massachusetts Roads dataset.
<p>
<img src='resources/tracking1.png' width='100%' />
<img src='resources/tracking2.png' width='100%' />
<img src='resources/tracking3.png' width='100%' />
<img src='resources/tracking4.png' width='100%' />
<img src='resources/tracking5.png' width='100%' />
<img src='resources/tracking6.png' width='100%' />
<img src='resources/tracking7.png' width='100%' />

# Code
The source code will be released here soon.
