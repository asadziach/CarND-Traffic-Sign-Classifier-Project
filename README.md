**Build a Traffic Sign Recognition Project**


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
[image13]: ./examples/softmax.png "softmax table"
[image14]: ./examples/graph.png "softmax table"
[image15]: ./examples/conv1_bias.png "conv1_bias.png"
[image16]: ./examples/conv1_weight.png "conv1_weight"
[image17]: ./examples/conv2_bias.png "conv2_bias"
[image18]: ./examples/conv2_weight.png "conv2_weight"
[image19]: ./examples/distribution.png "distribution"
[image20]: ./examples/scalars_conv1.png "scalars_conv1"
[image21]: ./examples/scalar_conv2.png "scalar_conv2"
[image22]: ./examples/accuracy.png "accuracy"
[image23]: ./examples/mean.png "mean"
[image24]: ./examples/xbar.png "xbar"
[image25]: ./examples/sigma.png "sigma"


### Data Set Summary & Exploration

I used the numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data unevenly distributed across classes. I used different colors for relative comparison of Training, Testing and Validation data set.

![alt text][image1]

### Design and Test a Model Architecture
#### Preprocessing
As a first step, I applied "Feature standardization". I linearly scaled image to have zero mean and unit norm. This was done by computing:

![alt text][image24]

where x bar is the average of all values in image, and

![alt text][image25]

sigma is the standard deviation of all values in image. It is capped away from zero to protect against division by 0 when handling uniform images.

I implemented it using TensorFlow that gets GPU acceleration. Here is an example of a traffic sign image before and after standardization.


![alt text][image2]

At first I was only grayscaling and normalizing but I was not happy with results. It turned out Standardization works better for this use case.

#### Augmentation

I decided to generate additional data. because you can notice in the first figure, there is lot of class imbalance. On one side few classes have north of 2k samples, but on the other hand couple of the classes have as low as 180 samples. Experience has shown that this skews final results.

To add more data to the the data set, I built a TensorFlow pipeline to perform the following operations in order:

1. Random Crop
1. Random Brightness
1. Random Saturation
1. Random Hue
1. Random Contrast
1. Standardization


Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is slight change in position and color.

I empirically narrowed down my augmented data pipeline to the above operations. I tried couple more techniques, like random flip and rotation but they did not have a positive outcome on results. Then there is question of how much data to be added. First I made all classes equal to the max class number, then I doubled all data 2x and 3x. The value of 2x gave me the best results.

Here is the data after augmentation, all class now have close to 4k samples.

![alt text][image4]

Number of training examples after augmentation = 172860

Samples per class = 4020

#### Architecture

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
| Fully Connected		| Input = 84. Output = 43					    |
| Softmax       		|                       					    |
 
![alt text][image14]

#### Training

| Parameter         		|     Description	        					|
|:---------------------:|:---------------------------------------------:| 
| Learning Rate   		| 0.001                							| 
| Epochs         		| 90						                 	| 
| Batch Size       		| 1024              							| 
| Optimizer         	| Adam Optimizer						        | 
| Dropout Input    		| 0.99              							| 
| Dropout Convolution	| 0.90              							| 
| Dropout Fully Connected| 0.5             				     			| 

                                                             , 
#### Approach to the solution

My final model results as calculated in cells 25, 26:
* training set accuracy of 1.000
* validation set accuracy of 0.973 
* test set accuracy of **0.963**
##### Precision
![alt text][image9]
##### Recall
![alt text][image10]
##### Fβ score
![alt text][image11]
##### Normalized Confusion Matrix
![alt text][image12]

I started with the basic version of Lenet introduced in the lab. The accuracy was less than .90. Lenet came out in 1998. A lot has been done since then. For example I added a Dropout layer between the two Fully Connected Nodes and my accuracy was more than .93, which is minimum required for this project.

I spent most of my time with the data preprocessing. I first tried grayscale. It was faster to train but not as accurate. Tried smoothing, did not work well. I started out with a simple normalization technique (x - 128.0) / 255.0 but eventually settled for Tensorflow based image standardization as mentioned in the first part of the writeup. I also tried histogram equalization with OpenCV and switching to HSV color space, it did not help.

First thing I noticed about some of the problematic classes was that, they had very little representation in the dataset and I was dealing with skewed classes. Initially I tried to double the data set but it did not help much. I tried to use tf.losses.softmax_cross_entropy() instead of tf.nn.softmax_cross_entropy_with_logits() to deal with class imbalance but I did not get much improvement. It seemed I needed to deal with 'Imbalanced Dataset' properly. I modified my code in cell 11, to have equal samples across all classes and my scores got a boost.

During data augmentation, I needed to do some perturbation with the original training data, otherwise I would be teaching nothing new to the CNN. I tried different combinations of cropping, rotation, flipping, color, brightness and contrast randomization. Eventually I settled down to the list mentioned in the start of this document.

All of the above was getting me to .941 score. As an experiment to improve further, I increased the depth of both my convolutional layers. I got to accuracy of 0.951. I noticed that now my validation accuracy was way up. I was overfitting. I introduced two more dropout layers and got my best results. I introduced the same dropouts to my earlier version of model which I call Lenet_Fast.

* The Fast version have test accuracy of 0.952 
* The Deeper model has test accuracy of **0.963**. 

#### Detailed look at Training process (TensorBoard)

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]

### Test a Model on New Images

####  Using my own German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image5] 


The first image might be difficult to classify because if you see in the confusion matrix above, the speed signs gets confused with each other. Given the fact that 30, 50, and 80 symbols are very similar in structure. Given a noisy training set and difference in font can throw the detections off.

#### 2. Detailed look at predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h       		| 30 km/h   									| 
| Stop      			| Stop  										|
| No Entry				| No Entry										|
| Right of way     		| Right of way					 				|
| Road Work 			| Road Work         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.3%.

#### 3. Top softmax probabilities for each image along with the sign type

The code for making predictions on my final model is located in the cell 41, 42 of the Ipython notebook.

The below image summarizes the results. Input image is on the leftmost column. Second row contains the highest detected probability and so forth. Softmax score for each is given on top of each image.

I made another observation that as I run deeper models for longer EPOCHs, the softmax scores for 2nd and 3rd prediction on the 50km/h sign gets lower. It indicates I either need to better my input data or to change my model if I hope to correctly classify my own 50km/h sign.

![alt text][image13]


### Visualizing the Neural Network 
Looking at the weights of trained NN, it is obvious that it has high level of activation for the sign edges. For speed signs, it is sensitive to the font edges. I found the No Entry sign interesting. It looks that some Feature Maps are sensitive to horizontal change dx, and some are sensitive to vertical change in pixels dy. Combining all these contribute to the overall shape detection.

![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
