# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[histogram_1]: ./writeup_images/visualization.png "Visualization - Histogram of images per class"
[histogram_2]: ./writeup_images/visualization_moredata.png "Histogram of images per class after adding additional data (via rotating other data)"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In the Jupyter notebook I calculated the size of the dataset (
* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32x32 (the pickle file is scaled down from their original size)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images were provided of each class type.

Looking at this histogram, it can be seen that a large number of classes have WAY less than the mean number of images and from testing we verified that this was leading to poor results on those classes as one would expect.

![original data histogram][histogram_1]

(see the next section for how I addressed this challenge)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To first address the "Stand Out Suggestions", the prior histogram revealed that the training data was very un-even and many classes had an insufficient amount of images.

To solve this problem, I created a minimum number of images (originally I used the mean which is around 750, but experimentation found that 500 works at least as well) and then generated images to fill out all of the undersized classes.  The extra images were created by repeatedly selecting a random image from the input data-set, then rotating it by +/- 5-degrees to 20-degrees (the angle and the sign were chosen randomly).

| ![original data histogram][histogram_1] | ![histogram of classes after adding additional images][histogram_2] |
|:---:|:---:|
| Initial dataset | After adding rotated images |

This made the data much more helpful on many classes. Since the additional data was based off of slight transforms of a very small amount of input-data, these classes will still be less robust than if they had real data, but this is a fairly effect stop-gap that can be a great alternative when it is not practical to collect additional data for training.

**Additional Preprocessing**

I tried several preprocessing methods and analyzed their effect to decide on whether to keep them.

**Grayscale** - Color is slightly useful for road signs when humans are interpreting them, but experimentally grayscaling the images consistently resulted in higher accuracy for the network.  This implies to me that the color (given the junkiness of original images) was actually serving to confuse the network more than help.  My hypothesis is that we're feeding raw pixel numbers to the CNN, whereas a human brain would itself be pre-processing and classifying the colors. If we see a red street-sign, regardless of the angle, the lighting conditions or a slight amount of fading due to age, we're going to see "red". Our input data could be seeing (255, 0, 0) vs (200, 100, 150) though. If we thought that color was extremely important, we could pre-process the input colors to "snap" the data to match a pallet of actual colors used in German street signs.

Grayscale cuts the image data in one-third (one channel instead of 3) so it should increase performance as well.

**Adaptive Histogram Equalization** - This adds contrast to the images. It was extremely time-prohibitive though and was more expensive than training the entire network. One workaround for this would have been to do this preprocessing once, then write all of the derived images out to a separate Pickle file and then load that on future runs. This would have been a bunch of work though, given that we were already at our target accuracy by the time this issue came up, I did not pursue this path further.

**Normalization** - This was one of the first things I tried, and I tried several types of normalization. I used the (data-128)/128 trick mentioned in the notebook, used normalization based on the actual mean and sum (these normalize from -1.0 to 1.0), and even normalized inside of a nice 0.1 to 0.9 box (which would prevent very high and very low values from being washed away or overpowered, and shockingly none of the normalizations resulted in accuracy improvements for me.

This is not consistent with what I would have expected, so I did some debugging to print out the raw pixel values before and after normalization and they DID appear in the ranges that I would have expected from normalization.  When I switched my normalization function to just be a NO-OP, accuracy was always the best.  This is a bit surprising for me and is one of the things I would return to investigate further if I ever needed to increase accuracy even more.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.




NOTE: CURRENTLY THIS FAR IN THE WRITEUP... GOING TO PULL SOME IMAGES INTO THE REPO THEN CONTINUE...











My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


