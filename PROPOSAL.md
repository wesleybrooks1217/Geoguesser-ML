
# geoguess-ml
![image](https://github.gatech.edu/storage/user/60157/files/a4fe5357-46df-4f2b-8462-99021752c736)
# Intro
This project involves the ability to discern the location of an image using machine learning. There has been a decent amount of work done in this field. In 2016, Google developed an AI called Planet which aimed to recognize the location of a photo anywhere in the world. At the University of Colorado Boulder, students aimed to create an AI which could determine the location of a photo in the continental U.S states(Theethira). However, this area of ML is far from being perfected. The Google Planet AI could only detect location with a country-accuracy of 28%(Brokaw). 
# Motivation
The motivation of this project is being able to geotag images without any additional information other than the pixels in the image. The ability to do this could have widespread uses, from tagging photos on social media sites to identifying the location of criminals or fugitives using only a small set of images. 
# Algorithms and Methods
For the proposed dataset, the Google Street API will be used to generate random street images across a selection of countries (Nguyen et al). Around 35,000 images will be generated to ensure the proposed model will have enough data to hypertune parameters and ensure accuracy with the validation set in subsequent trials. The dataset will be the input for a convolutional neural network implemented through PyTorch libraries and packages. We will normalize the data and then pass it through a CNN with convolutional, ReLU, max pooling, and fully connected layers (Swapna). Finally, we will use a cross entropy loss function and optimize with gradient descent.

![general CNN architecture](https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1200%2C564&ssl=1)
*Similar CNN Architecture*   [source](https://developersbreach.com/convolution-neural-network-deep-learning/)
# Discussion and Expected Results
There will be a few constraints put on our results. We will not consider the exact location of the image. Instead, we will consider if the correct country was chosen. In other predictive uses of convolutional neural networks, well-developed CNNs have been able to outperform humans by around 33 percent (Mrázová et al). Accounting for the unique characteristics of our project in addition to knowledge/time constraints, we hypothesize that our algorithm will provide correct predictions at a rate slightly greater than an educated human.
In order to quantify the overall accuracy of our model, we will use an accuracy score function which computes what percentage of predictions are correct. As a measure of how well our model matches our hypothesis, we will compare the accuracy of our algorithm and the accuracy of a human participant in geotagging a set of images. Our algorithm will be successful if it produces correct guesses at a rate greater than or a participant.
# References
 1. Brokaw, Alex. “Google's Latest AI Doesn't Need Geotags to Figure out a Photo's Location.” The Verge, The Verge, 25 Feb. 2016, https://www.theverge.com/2016/2/25/11112594/google-new-deep-learning-image-location-planet. 
2. Nguyen, Q. C., Huang, Y., Kumar, A., Duan, H., Keralis, J. M., Dwivedi, P., Meng, H.-W., Brunisholz, K. D., Jay, J., Javanmardi, M., &amp; Tasdizen, T. (2020, September 1). Using 164 million google street view images to derive built environment predictors of COVID-19 cases. MDPI. Retrieved October 7, 2022, from https://www.mdpi.com/1660-4601/17/17/6359  
3. I. Mrázová, J. Pihera and J. Velemínská, "Can N-dimensional convolutional neural networks distinguish men and women better than humans do?," The 2013 International Joint Conference on Neural Networks (IJCNN), 2013, pp. 1-8, doi: 10.1109/IJCNN.2013.6707101.  
Swapna. “Convolutional Neural Network: Deep Learning.” Developers Breach, 25 Jan. 2022, https://developersbreach.com/convolution-neural-network-deep-learning/#1-1-
  convolution.  
4. Popa, Bogdan. “Music Video Created with Google Maps Street View Images Is Surprisingly Cool.” Autoevolution, 16 May 2020, https://www.autoevolution.com/news/music-video-created-with-google-maps-street-view-images-is-surprisingly-cool-143752.html#agal_0. 
5. Theethira, Nirvan S. “GEOGUESSR AI: IMAGE BASED GEO-LOCATION.” GEOGUESSR AI, Mar. 2020, https://nirvan66.github.io/geoguessr.html. 
# Gantt Chart
[GanttChart.xlsx](https://github.gatech.edu/dgorin6/geoguess-ml/files/1193/GanttChart.xlsx)
