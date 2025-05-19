# cs0451-pneumonia-detection
By Cameron Hudson, Robsan Dinka, Lia Smith, and Emmanuel Towner

## Abstract
[Our-Project](https://github.com/EpicET/cs0451-pneumonia-detection)
Within our blog post, we created a neural network and implemented three different binary classifers trained on chest x-ray image data to detect pneumonia based on images. We used convolution layers to convert images into latent vectors by which we could feed into our various machine learning models: a transformer, an SVM, and a gradient boosting model. Through analyzing the accuracy of each model, we discovered a similar accuracy between the models of around 78% on testing data.

## Introduction
Within our project, we wanted to compare 3 seperate binary classification machine learning models, seeing which is best for an image classification task. Our project attempts to uncover what types of algorithms are best for binary image classification tasks using the pneumonia chest xray dataset, with our models being trained to discern pneumonia based on chest xray images only. This dataset demonstrates a case where finding the most optimal image classifcation algorithm is very important as it could result in saving a life. Our research could also inform which types of algorithms should be considered other important image classification tasks. Within [MobileNet Pneumonia Classification (2023 study)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10252226/), the researchers mainly focus on deep learning algorithms to tackle this same image classification task. Through their research, they discovered the MobileNet CCN gave the best accuracy on two datasets with values of 94.23% and 93.75%. In another study titled [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225), researchers create their own CNN known as CheXNet that detects pneumonia as well as other chest related illnesses (fibrosis, hernia, etc.) that which accuracies ranging from 0.7 to 0.9. With such a large focus on CNNs for this image classification, ww wanted to determine if other kinds of algorithms good for binary classifcation could also be useful image classifiers.  

## Values Statement
The potenital users of our project would be primary care clinicians and radiologists who must regularly discern chest related illnesses through x-rays. These machine learning models trained on chest x-ray image data may help them make more informed desicions if they are trying to discern specifically pneumonia. 

I believe that our work contributes to AI researchers who are studying how to optimize for proformance in image classifcation tasks especially regarding medical concerns. If it can inform medical researchers on what machine learning models are best at medical image classification, they and their patients can also benefit from greater accuracy in detecting chest related illnesess.

Because our models are quite poor at predicting images without pneumonia correctly, they could falsely flag patients as having pneumonia, which may lead them to incur unnessecary medical expenses. Based on the background of these patients, this could seriously affect patients who struggle financially. 

Our group personally enjoyed and had an interest in each of the algorithms that we worked on and took this project as a learning experience to expand our knowledge on what image vectorization and binary classification algorithms are out there and how they differ from what we have learned through our class assignments.

Based on our experiements, we believe if our project can help inform image classification tasks, especially those in the medical field, then the world can become a better place by being able to help people detect illnesses earlier and possibly save lives. 

## Materials

Our data comes from the Pneumonia Chest X-ray dataset in Kaggle. This data came from the Guangzhou Women and Children’s Medical Center. Samples were collected from patients and labels were created by pneumonia specialists, with two specialists making labels and then a third corroborating the label of normal or pneumonia. Our data lacks information regarding the severity or time-span of the pneumonia for positive cases, meaning that the model has no clear way of understanding which x-rays should be encoded closer or further away from the normal cases. Additionally, the dataset has a 64% / 36% split, with the majority of xrays containing positive cases of pneumonia. This bias happens to work out well for mitigating false negatives, however it makes models have more difficulty understanding when an xray is normal. 

## Methods
In our convolutional neural network for embedding the chest xrays, we utilized a variational autoencoder with contrastive loss. The model projects the vectors onto a 64 dimensional latent space through a gaussian disstribution, with KL-divergence, Contrastive loss, and cross entropy loss. The contrastive learning gives the latent dimensions more direction as to how negative pairs (pairs of data with different labels) should be encoded further apart in the data, while positive pairs should be encoded more similarly. The convolutional layers in the model are used to capture important spatial features from the images. After the images are vectorized, we use a Support Vector Machine, XGBooste, and Transformer to then classify the latent vectors as having pneumonia or being normal. We choose these binary classification models, since the latent space contains complex non-linear trends. All of these models are exceptional at handling non-linear data well. The image embedder was trained in batches of 32 images each. It was trained in google collab in order to take advantage of the GPU. This data was split into a test and train divsion. 5100 images were contained in the training data and 620 were contained in the test dataset. The test and training dataset allowed us to determine if the model was generalizing well to new data without taking too many data points from the training loop. Additionally, fivefold cross validation was employed to prevent overfitting on the test data. While auditing the models, we discovered that our models got between 75-78% accuracy, with extremely high precision rates of 86-93% and lower recall rates of 37-41%. This fits with the cost imbalance associtated with a false positive versus a false negative diagnosis. We traded a bit of accuracy in order to capture the majority of positive cases. 

## Results 
<img width="262" alt="image" src="https://github.com/user-attachments/assets/2bd2c6ef-4d1a-4994-ad37-e454924eee0d" />
<img width="463" alt="image" src="https://github.com/user-attachments/assets/897d1e6c-110d-4291-bec7-f3ea6cc16fc5" />

As demonstrated before, the models contained much higher precision rates than recall in order to catch more of the positive pneumonia cases due to their costliness as compared to the costs associated with missing a normal case. Within the models, The transformer did the best, with the highest Recall and Precision of 93% and 41% respectively. The F-1 Score of 57% suggests that the model was beginning to learn differences between the classes but still encountered much difficulty. This is also present the the 3-D PCA plot of the latent vectors where it becomes evident that many of the embeddings are caught in an overlapping region where both classes meet. The results suggest that the image embeddings need more fine tuning to increase accuracy and recall. 


## Conclusions

The project accomblished many of the goals that we set out to accomblish during the duration of this project and also failed to meet others. We got a working convolutional neurlal network to embed the images and learn important features of those images. We correctly identify 93% of all pneumonia cases. On the other hand, We correctly identify less than half of all normal cases. This project demonstrates the difficulty of complex machine learning tasks without good computational resources. Running and auditing the cnn alone takes two hours per run with a GPU. Due to this contraint, we were unable to readily take advantage of all of the data available. Additionally, the binary classifcation models also took 5-15 minutes depending on the model. The most apparent hurdle in this project was creating a complex model while also being able to run it in a reasonable amount of time. Other pneumonia binary classification projects are able to get higher accuracy through the usage of premade Resnet models. These models are trained on millions of images and use residual connections to improve the performance of neural networks. If we had more time, we would do a more thorough error analysis of misclassified normal images to understand what features the model is missing and improve the architecture to capture that feature. Additionally, we would utilize more of the training data without run-time constriants and try adopting residual neural network arhitecture to improve performance.

## Group Contributions:
Emmanuel: Set up most of the Github, worked on the Transformer
Cameron: Worked on the Introduction, Abstract, and Values statement of the Blog Post in addition to the Support Vector Machine
Robsan: Worked on the XGBooste Model
Lia: Worked on evaluation metrics for the models, embedding the images, conclusions, results, methods, materials, and group contributions. 







