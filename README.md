# SDCND-P2
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./fig_for_writeup/sample.png 
[image2]: ./fig_for_writeup/histogram.png
[image3]: ./fig_for_writeup/normalized.png 
[image4]: ./fig_for_writeup/gray-normalized.png 
[image5]: ./fig_for_writeup/graph.png 
[image6]: ./fig_for_writeup/featuremap.png
[image7]: ./fig_for_writeup/new.png 
 

## Rubric Points and their discussion

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 
I used mainly the numpy library calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First the code is displaying a sample dataset accompagnied by its right label on each image, also an exploratory histogram was drawn to calculate the number of training samples in each calss. 

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* Concerning preprocessing: I tried two approaches, the first is normalization only and the second is grayscaling and then normalization, the one used for the submission is the second as it showed better results over the testing and the new images. 
  below is a picture for the normalized colored images
  ![alt text][image3]
  
  and a picture of grayscaling+normalization with the corresponding labels
  ![alt text][image4]
  
* Note For data augmentation: as realized from the histogram above, some classes have few training examples which might cause defects in the training process, however data augmentation is not performed in this code, but a suggestion for it would be to augment the data of the classes with less examples after editing the existing samples randomly to balance the training data.
  
// Noting also: that in the new images chosen I was keen to include some samples of the classes with few training examples such as (25,37,38) and as the classifier performed well with those I did not perform augmentation.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is like the LeNet lab solution:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 5x5     	| 1x1 stride, padding, outputs 28x28x6	|
| RELU					|		activation										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5 	    | 1x1 stride, padding, outputs 10x10x16		|
| RELU					|	activation											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Fully connected | input 400 output 120|
|  RELU					|					activation							|
|	Dropout | keep probability 0.5 |
|	Fully connected | input 120 output 84|
|  RELU					|			activation								|
|	Dropout | keep probability 0.5 |
|	Fully connected | input 84 output 43|

 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* optimizer: ADAM optimizer
* Batch size: 100
* Number of Epochs: 15
* Learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.957
* test set accuracy of 0.932
* mu= 0 and sigma= 0.1 (mean and standard deviation to generate random weights)

LeNet architecture was chosen for this project: 
* I thought it would be relevant as it contains 2 convolution layers to enhance feature extraction and recognition of the images each accompanied with RELU activation to add non linearity to the model, dropout layer to reduce overfitting and fully connected layers to combine the feature from previous layers. 
* In order to monitor whether the model is overfitting in other words the gap between the validation and training accuracy is enlarging or not over epochs the following graph of both accuracies was plotted, and we can notice that the gap is relatively stable which proves that the model is working okay

![alt text][image5]
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web in which Classes with ew example numbers are included (25,37,38)
![alt text][image7]

They were grayscaled and normalized first after classification

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Class number
|:---------------------------------------:|:---------------------------------------------:|:----:| 
| Road work      		| Road work    									| 25|
| Go straight or left    			| Go straight or left  								|37|
| General Caution				| General Caution										|18|
|Right-of- way at the next intersection	      		| Right-of- way at the next intersection					 				|11|
| Speed limit 80		|  Speed limit 80    							|5|
| Keep right		| Keep right	   							|38|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
, .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


