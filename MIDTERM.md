
# geoguess-ml
![image](https://github.gatech.edu/storage/user/60157/files/a4fe5357-46df-4f2b-8462-99021752c736)
# Intro
This project involves the ability to discern the location of an image using machine learning. We have made significant progress since our last report. Firstly, we have used the Google StreetView API to generate a large dataset of images, and we were able to check attributes of the data that we recieved in order to avoid noise in our dataset. Secondly, using a collection of PyTorch libraries, we implemented principal component analysis (PCA) and a convolutional neural network (CNN) in order to make predictions on each image. 
# Motivation
The motivation of this project is being able to geotag images without any additional information other than the pixels in the image. The ability to do this could have widespread uses, from tagging photos on social media sites to identifying the location of criminals or fugitives using only a small set of images. 
# Data Collection Methods
We utilized the Google StreetView API to generate 20,000 street images from a selection of ten countries. Specifically, we got 2,000 images from each of the United States, Russia, Great Britain, South Africa, Argentina, Japan, Australia, Portugal, Israel, and Kenya. Our intention with this selection of countries was to have enough geographic diversity to train the model on various features/characteristics across the world, while also maintaining some countries with similar features (e.g. South Africa and Kenya) to examine how accurately the model differentiated between these countries. While it is difficult to get StreetView data from an individual country directly, we were able to use the country's latitude and longitude bounds to reduce the areas where the API had to search, and verify data about each image we recieved in this range to ensure it was in our intended country. Using this, we were able to obtain a clean dataset that we could use to implement our model. Shown below are two of our images, from Kenya and Japan respectively.
![KEN 1296](https://github.gatech.edu/storage/user/63747/files/4d0cab57-e065-4d5d-aa0a-4444d0d5db33)
![JPN 1297](https://github.gatech.edu/storage/user/63747/files/672a20f3-2467-475a-b9b0-c514bcff5b6d)
 
 





# Algorithms and Methods
For dimensionality reduction, we used the PCA algorithm from the scikit-learn module, and reduced the number of features present to 20 (this is a hyperparameter that we can modify for our final report) for every image, after which we used a pandas dataframe to store all of the image vectors and the actual values.  We then input the modified image data into a CNN implemented through PyTorch libraries and packages. Our forward pass algorithm sends the data through three 2-D convolutional layers and three linear layers. Within these layers, we use the rectified linear unit function (ReLU); the advantage of this is that it allows for zero values and is a linear function, allowing for a faster, more sparse representation of data than would be used in a sigmoid function. After using ReLU on the convolution layers, we use max pooling, which down-samples images by applying a max filter to every 2 x 2 nonoverlapping submatrix. After the forward pass, we developed a backpropagation algorithm that utilized a cross entropy loss function, and used gradient descent to update the parameters/inputs. We can modify the number of epochs that we run this algorithm as another hyperparameter of our model. 

![general CNN architecture](https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1200%2C564&ssl=1)
*Similar CNN Architecture*   [source](https://developersbreach.com/convolution-neural-network-deep-learning/)
# Discussion and Expected Results
There will be a few constraints put on our results. We will not consider the exact location of the image. Instead, we will consider if the correct country was chosen. In other predictive uses of convolutional neural networks, well-developed CNNs have been able to outperform humans by around 33 percent (Mrázová et al). Accounting for the unique characteristics of our project in addition to knowledge/time constraints, we hypothesize that our algorithm will provide correct predictions at a rate slightly greater than an educated human.
In order to quantify the overall accuracy of our model, we will use an accuracy score function which computes what percentage of predictions are correct. As a measure of how well our model matches our hypothesis, we will compare the accuracy of our algorithm and the accuracy of a human participant in geotagging a set of images. Our algorithm will be successful if it produces correct guesses at a rate greater than or a participant.

# Results and Discussion
Before training the model, we performed feature reduction by applying PCA to keep the 20 most prevalent features for each image. After running PCA, we trained the model over 10 epochs and received a training accuracy of 96% for the final epoch.

![Epoch Training Data](Training_Accuracy.png)

Once the model was trained, we tested the model with a total of 500 images and received a testing accuracy of 28%.

![Country Labels](Labels.png)
![ML Metrics](Analysis.png)

We also analyzed the predictive performance of each feature by calculating the percision, recall, and F1 for each label and received the following metrics:


Our metrics indicate a high predictive performance for Japan, South Africa, and Portugal while a low predictive performance for Australia, United States, and Russia. These results are expected, as Japan, South Africa, and Portugal are smaller countries which led to more similar images being collected. On the other hand, Australia, United States, and Russia are larger countries leading to less similar images being collected. The difference between the similarity of the images collected affect the model's accuracy for predicting each individual label. Our model also indicates overfitting, which we will attempt to improve on by tuning our hyperparameters for the CNN.

# Gantt Chart (Including Contribution Table)
[GanttChartFinal.xlsx](GanttChartFinal.xlsx)

