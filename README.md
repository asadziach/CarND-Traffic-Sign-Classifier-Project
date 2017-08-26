**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/standardization.png "Standardization"
[image3]: ./examples/augmented.png "Augmented Example"
[image4]: ./examples/augmented_bar.png "Augmented Bar"
[image5]: ./examples/my_own_all.png "Traffic Sign 2"
[image6]: ./examples/v1.png "Input"
[image7]: ./examples/v2.png "Conv1 Visualization"
[image8]: ./examples/v3.png "Conv2 Visualization"
[image9]: ./examples/precision.png "Precision"
[image10]: ./examples/recall.png "Recall"
[image11]: ./examples/f_beta.png "F Beta"
[image12]: ./examples/confusion_matrix.png "Confusion Matrix"

## Rubric Points

### Data Set Summary & Exploration

I used the numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data unevenly distributed acress classes. I used different colors for relative comparision of Training, Testing and Validation data set.

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I applied "Feature standardization". I linearly scaled image to have zero mean and unit norm. This was done by computing:

$$\acute{x} = \frac{{x}-\bar{x}}{\acute{\sigma}}$$
where x bar is the average of all values in image, and

$$\acute{\sigma} = max(\sigma, \frac{1}{\sqrt{size}})$$

sigma is the standard deviation of all values in image. It is capped away from zero to protect against division by 0 when handling uniform images.

I implemented it using TensorFlow that gets GPU acceleration. Here is an example of a traffic sign image before and after standardization.


![alt text][image2]

At first I was only grayscaling and normalizing but I was not happy with results. It turned out Standardization works better for this use case.

I decided to generate additional data. because you can notice in the first figure, there is lot of class imbalance. On one side few classes have north of 2k samples, but on the other hand couple of the classes have as low as 180 samples. Experince has shown that this skews final reuslts. Deep learing algorithem tends to favor the more frequent classes.  

To add more data to the the data set, I built a Tensorflow pipeline to perform the following operations in order:

1. Random Crop
1. Random Brightness
1. Random Saturation
1. Random Hue
1. Random Contrast
1. Standardization


Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is slight change in position and color.

I empirically narrowed down my augmented data pipeline to the above operations. I tried couple more techniques, like random flip and rotation but they did not have a positve outcome on results. Then there is question of how much data to be added. First I made all calsses equal to the max class number, then I doubled all data 2x and 3x. The value of 2x gave me the best numbers.

Here is the data after augmentation, all class now have close to 4k samples.

![alt text][image4]

Number of training examples after augmentation = 172860

Samples per class = 4020

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Dropout         		| 0.99 training only							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout         		| 0.90 during training 							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Fully connected		| Input = 1600. Output = 120					|
| RELU					|												|
| Fully connected		| Input = 120 Output = 84   					|
| RELU					|												|
| Dropout         		| 0.5 during training 							|
| Fully connected		| Input = 84. Output = 43					    |
| Softmax       		|                       					    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Paramter         		|     Description	        					|
|:---------------------:|:---------------------------------------------:| 
| Learning Rate   		| 0.001                							| 
| Epochs         		| 90						                 	| 
| Batch Size       		| 1024              							| 
| Optimizer         	| Adam Optimizer						        | 
| Dropout Input    		| 0.99              							| 
| Dropout Convolution	| 0.90              							| 
| Dropout Fully connnected| 0.5             							| 

                                                             , 
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were as calculated in cells 23,24:
* training set accuracy of ???
* validation set accuracy of 0.973 
* test set accuracy of **0.963**
##### Preciesion
![alt text][image9]
##### Recall
![alt text][image10]
##### Fβ score
![alt text][image11]
##### Confusion Matrix
![alt text][image12]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I started with the based verstion of Lenet introduced in the lab. The accuracy was less than .90. Lenet came out in 1998. A lot has been done since then. For example I added a Dropout layer between the two Fully Connected Nodes and my accuracy was more than .93, which is miminum required for this project.
I spent most of my time with the data preprocessing. I tried grayscale. It was faster to train but not as accurate. Treid smoothing. I started out with a simple normalization techinque (x - 125.0) / 255.0 but enentually sattled for Tensorflow based image standardization as mentioned in the first part of the writeup. I tried histogram equalization with OpenCV and swiching to HSV colorspace, it did not help.

First thing I noticed about some of the problamtic classes was that, they had very little reppresentation in the dataset and I was dealing with skewed classes. Initially I tried to double the data set but it did not help much. I tried to use tf.losses.softmax_cross_entropy insated of tf.nn.softmax_cross_entropy_with_logits to deal with class imbalance but I did not get much improvement. It seemed I need to deal with Imbalanced Dataset properly. I modified my code to have equal samples across all classes and my scores got a boost.

During data agumentation, I needed to do some perturbation with the orginal training data, othwise I would be teaching nothing new to the CNN. I tried different combos of cropping, rotation, flipping, color, brightness and contrast randomization. Eventually I satled down to the list mentioned in the start of this document.

All of above was getting me to .941 score. As an experiment to improve futher, I inceased the depth of both my convolution layers. I got to accuracy of 0.951. I noticed that now my validtion accuracy was way up. I was overfitting. I intorduced two more dropout layers and it brought to my current result of 0.963. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 


The first image might be difficult to classify because if you see in the confusion matrix above, the speed signs gets confused with each other. Given the fact that 30, 50, and 80 symobls are very similar in structure. Given a noisy training set and difference in font can throw the detections off.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h       		| 30 km/h   									| 
| Stop      			| Stop  										|
| No Entry				| No Entry										|
| Right of way     		| Right of way					 				|
| Road Work 			| Road Work         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... ???

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
