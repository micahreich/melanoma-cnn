# Convolutional Neural Network Melanoma Detection Model

Convolutional Neural Network Melanoma Detection Model is a repository which contains a deep learning classifier trained to classify images of skin moles as malignant, benign, or indeterminant. 

## Installation
This repository depends on the following (clone into a new folder with a python virtual environment):

[Tensorflow 1.13.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1)

[ISIC-Archive-Downloader](https://github.com/GalAvineri/ISIC-Archive-Downloader)

The contents of this repository can all be installed in one location. Please refer to the following project structure for information regarding each file and directory in the repo:

    .
	├── Dataset/
		├── Descriptions/    # Contains metadata and labels for training images
		├── Images/    # Contains training images
    ├── model.py    # Contains code for structure of AlexNet CNN model
    ├── preprocessing.py    # Contains code for data preprocessing to be fed into AlexNet
    ├── train.py     # CNN training pipeling
    └── README.md


## Usage
Navigate to your python virtual environment where you have cloned the repository and loaded the images using ```ISIC-Archive-Downloader``` and run the training script:

```bash
python train.py
```
That's it! Training should begin; training times will be dependent upon the user's GPU or CPU capabilities

## Neural Network Structure
The model's structure follows that of the AlexNet CNN classifier structure defined in "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Geoffrey E. Hinton et al:

| Order  | Layer Title | Layer Dimensions  | 
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

## Attributions
[AlexNet Paper by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[ISIC-Archive-Downloader by Oren Talmor and Gal Avineri](https://github.com/GalAvineri/ISIC-Archive-Downloader)
