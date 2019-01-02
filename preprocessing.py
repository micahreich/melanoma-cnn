import os
from sklearn.preprocessing import LabelBinarizer
import PIL
import pickle
import numpy as np
from PIL import Image


def get_data(training):
    new_width = 227
    new_height = 227

    images_arr = []
    labels = []

    for filename in os.listdir("Data/Descriptions"):
        image = Image.open(
            "Data/Images/" + filename + ".jpeg")
        image = image.resize((new_width, new_height), PIL.Image.ANTIALIAS)

        images_arr.append(np.array(image))

    for filename in os.listdir("Data/Descriptions"):
        f = open("Data/Descriptions/" + filename)
        for i, line in enumerate(f):
            if "benign_malignant" in line:
                if line[27] == "b":
                    labels.append("benign")
                elif line[27] == "m":
                    labels.append("malignant")
                else:
                    labels.append("")

    print("Images of this dataset are of shape: ", images_arr[0].shape)

    encoder = LabelBinarizer()
    transfomed_labels = encoder.fit_transform(labels)
    print(transfomed_labels[0])
    #   80% of data used for training, 20% used for testing
    train_imgs = images_arr[:int(len(images_arr * 0.8))]
    train_labels = transfomed_labels[:int(len(transfomed_labels * 0.8))]
    test_imgs = images_arr[int(len(images_arr * 0.2)):]
    test_labels = transfomed_labels[int(len(transfomed_labels * 0.2)):]

    if training:
        tr_im_filename = 'training_images'
        tr_im_outfile = open(tr_im_filename, 'wb')
        pickle.dump(train_imgs, tr_im_outfile)
        tr_im_outfile.close()

        tr_lb_filename = 'training_labels'
        tr_lb_outfile = open(tr_lb_filename, 'wb')
        pickle.dump(train_labels, tr_lb_outfile)
        tr_lb_outfile.close()

        print(str(len(train_imgs)) + " Training Images in dataset")
        print(str(len(train_labels)) + " Training labels in dataset")

        return train_imgs, train_labels
    else:
        ts_im_filename = 'testing_images'
        ts_im_outfile = open(ts_im_filename, 'wb')
        pickle.dump(train_imgs, ts_im_outfile)
        ts_im_outfile.close()

        ts_lb_filename = 'testing_labels'
        ts_lb_outfile = open(ts_lb_filename, 'wb')
        pickle.dump(train_labels, ts_lb_outfile)
        ts_lb_outfile.close()

        print(str(len(test_imgs)) + " Training Images in dataset")
        print(str(len(test_labels)) + " Training labels in dataset")
        return test_imgs, test_labels


get_data(True)
