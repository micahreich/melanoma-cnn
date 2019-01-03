# Cancerous Mole Detection CNN
*A convolutional neural network built with Tensorflow which classifies moles as malignant, benign or indeterminate based on an image*

## 😃 Introduction
This project serves as a demonstration of how machine learning and deep neural networks can achieve high accuracies in medical-classification. Specifically, the data used in this model was the International Skin Imaging Collaboration's (ISIC) dataset of ~25,000 hand-labeled images of benign and malignant skin moles.

## 🧠 Deep Neural Networks
Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.
(Skymind.ai)

## 📄 Dataset
As mentioned above, this model was trained with the goal of classifying images of moles as benign, malignant or indeterminant. More information about the ISIC and their numerous datasets can be found at their website, https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main.

The images included in the full download of their images are of varying sizes, mostly 768 by 1024 pixels. Each color image has 3 RGB color channels and is in .jpeg format. In order to simplify the amount of data processed, the model resizes the images to the shape of 227x227x3. An example image:

<img src="https://i.imgur.com/suPBjlb.jpg" height="227" width="227"/>

## ⛩ Architecture
The architecture of this model resembles that of AlexNet, a high performing convolutional neural network architecture on ImageNet's classification challenge. The model used in this project does not exactly emulate the AlexNet architecture. The layers used are outlined below:

| Order  | Layer Title | Information  | 
| ------------- | ------------- |-------|
| 1  | 2D Convolutional Layer w/ ReLU activation  | 96 filters of size 11 x 11  with a stride of 4 |
| 2  | 2D Maxpooling Layer  | 3x3 kernel size with a stride of 2 |
| 3  | 2D Convolutional Layer w/ ReLU activation  | 256 filters of size 5 x 5  with a stride of 1 |
| 4  | 2D Maxpooling Layer  | 3x3 kernel size with a stride of 2 |
| 5  | 2D Convolutional Layer w/ ReLU activation  | 384 filters of size 3 x 3  with a stride of 1 |
| 6  | 2D Convolutional Layer w/ ReLU activation  | 384 filters of size 3 x 3  with a stride of 1 |
| 7  | 2D Convolutional Layer w/ ReLU activation  | 256 filters of size 3 x 3  with a stride of 1 |
| 8  | 2D Maxpooling Layer  | 3x3 kernel size with a stride of 2 |
| 9  | Fully Connected Layer | 4096 neurons & outputs |
| 10  | Fully Connected Layer | 4096 neurons & outputs |
| 11  | Output Layer w/ Softmax activation | 3 output neurons |

## ✅ To Do
Below are some changes I will implement which will help the performance of the model and it's general efficiency
- [ ] Use Tensorflow's "Dataset" module to feed data into training rather than feed_dict
- [ ] Implement normalization layers in model

## 👏 Attributions
AlexNet paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Medium Post by Hao Gao: https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637

A Helpful Tool for Downloading the ISIC Dataset: https://github.com/GalAvineri/ISIC-Archive-Downloader

