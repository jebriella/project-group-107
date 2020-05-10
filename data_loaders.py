import csv
import os
from random import shuffle
import random
import numpy as np
from skimage.io import imread
import math
import tensorflow as tf

def dataloader():

    img_list = []
    label_list = []
    path = 'data/Dataset'

    #'data/Training_dataset.csv'

    with open('data/Training_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            img_list.append(row[0])
            label = np.zeros(4)
            for i in range(1,5):
                if row[i] == '1.0':
                    label[i-1] = 1
                    break
            label_list.append(label)

    data = list(zip(img_list, label_list))
    shuffle(data)
    img_list[:], label_list[:] = zip(*data)

    total = len(img_list)
    train_nr = math.ceil(0.7*total)

    X = np.zeros((train_nr, 256, 256, 3), dtype=np.float32)
    y = np.zeros((train_nr, 4), dtype=np.float32)

    X_val = np.zeros((total-train_nr, 256, 256, 3), dtype=np.float32)
    y_val = np.zeros((total-train_nr, 4), dtype=np.float32)

    for i in range(total):
        #print(i)
        img = imread(os.path.join(path, img_list[i] + ".jpg"))
        img = img/256

        if i < train_nr:
            X[i,:,:,:] = img
            y[i,:] = label_list[i]
        else:
            X_val[i-train_nr,:,:,:] = img
            y_val[i-train_nr,:] = label_list[i]

    return X, y, X_val, y_val


def dataloader_oversample():

    img_list = []
    label_list = []
    path = 'data/Dataset'

    #'data/Training_dataset.csv'

    with open('data/Training_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            img_list.append(row[0])
            label = np.zeros(4)
            for i in range(1,5):
                if row[i] == '1.0':
                    label[i-1] = 1
                    break
            label_list.append(label)

    data = list(zip(img_list, label_list))
    shuffle(data)
    img_list[:], label_list[:] = zip(*data)

    total = len(img_list)
    train_nr = math.ceil(0.7*total)

    t_img = img_list[0:train_nr]
    t_label = label_list[0:train_nr]

    val_img = img_list[train_nr:total]
    val_label = label_list[train_nr:total]

    for i in range(train_nr):
        if t_label[i][3] == 1:
            t_img.append(t_img[i])
            t_label.append(t_label[i])

            t_img.append(t_img[i])
            t_label.append(t_label[i])

            t_img.append(t_img[i])
            t_label.append(t_label[i])

    for i in range(len(val_img)):
        if val_label[i][3] == 1:
            val_img.append(val_img[i])
            val_label.append(val_label[i])

            val_img.append(val_img[i])
            val_label.append(val_label[i])

            val_img.append(val_img[i])
            val_label.append(val_label[i])

    data = list(zip(t_img, t_label))
    shuffle(data)
    t_img[:], t_label[:] = zip(*data)

    train_nr = len(t_img)
    val_nr = len(val_img)

    X = np.zeros((train_nr, 256, 256, 3), dtype=np.float32)
    y = np.zeros((train_nr, 4), dtype=np.float32)

    X_val = np.zeros((val_nr, 256, 256, 3), dtype=np.float32)
    y_val = np.zeros((val_nr, 4), dtype=np.float32)

    for i in range(train_nr):
        #print(i)
        img = imread(os.path.join(path, t_img[i] + ".jpg"))
        img = img/256

        X[i,:,:,:] = img
        y[i,:] = t_label[i]


    for i in range(val_nr):
        #print(i)
        img = imread(os.path.join(path, val_img[i] + ".jpg"))
        img = img/256

        X_val[i,:,:,:] = img
        y_val[i,:] = val_label[i]

    return X, y, X_val, y_val


def augment(img,channel_shift):
    aug = tf.keras.preprocessing.image.random_rotation(img, 360, row_axis=0, col_axis=1, channel_axis=2, fill_mode='wrap', cval=0.0, interpolation_order=1) #rotation between 0 and 360 degrees
    aug = tf.keras.preprocessing.image.random_zoom(aug, (0.6,1), row_axis=0, col_axis=1, channel_axis=2, fill_mode='wrap', cval=0.0, interpolation_order=1) #zoom in
    aug = np.flip(aug, random.randint(0,1)) #flip over x or y-axis
    aug = tf.keras.preprocessing.image.random_shear(aug, 20, row_axis=0, col_axis=1, channel_axis=2, fill_mode='wrap',cval=0.0, interpolation_order=1)
    aug = aug/256
    if channel_shift == '1':
    aug = tf.keras.preprocessing.image.apply_channel_shift(aug, -0.3, channel_axis=2)
    elif channel_shift == '2':
    aug = tf.keras.preprocessing.image.apply_channel_shift(aug, 0.1, channel_axis=2)

    return aug


def dataloader_augmented():
    img_list = []
    label_list = []
    path = 'data/Dataset'

    #'data/Training_dataset.csv'

    with open('data/Training_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            img_list.append(row[0])
            label = np.zeros(4)
            for i in range(1,5):
                if row[i] == '1.0':
                    label[i-1] = 1
                    break
            label_list.append(label)

    data = list(zip(img_list, label_list))
    shuffle(data)
    img_list[:], label_list[:] = zip(*data)

    total = len(img_list)
    train_nr = math.ceil(0.7*total)

    t_img = img_list[0:train_nr]
    t_label = label_list[0:train_nr]

    val_img = img_list[train_nr:total]
    val_label = label_list[train_nr:total]

    for i in range(train_nr):
        if t_label[i][3] == 1:
            t_img.append(t_img[i]+'.1')
            t_label.append(t_label[i])

            t_img.append(t_img[i]+'.2')
            t_label.append(t_label[i])

            t_img.append(t_img[i]+'.3')
            t_label.append(t_label[i])

    for i in range(len(val_img)):
        if val_label[i][3] == 1:
            val_img.append(val_img[i]+'.1')
            val_label.append(val_label[i])

            val_img.append(val_img[i]+'.2')
            val_label.append(val_label[i])

            val_img.append(val_img[i]+'.3')
            val_label.append(val_label[i])


    data = list(zip(t_img, t_label))
    shuffle(data)
    t_img[:], t_label[:] = zip(*data)

    train_nr = len(t_img)
    val_nr = len(val_img)

    X = np.zeros((train_nr, 256, 256, 3), dtype=np.float32)
    y = np.zeros((train_nr, 4), dtype=np.float32)

    X_val = np.zeros((val_nr, 256, 256, 3), dtype=np.float32)
    y_val = np.zeros((val_nr, 4), dtype=np.float32)

    for i in range(train_nr):
        #print(i)
        img_name = t_img[i]
        if img_name[len(img_name)-2]=='.':
            channel_shift = img_name[len(img_name)-1]
            img_name = img_name[:-2]
            img = imread(os.path.join(path, img_name + ".jpg"))
            img = augment(img,channel_shift)
        else:
            img = imread(os.path.join(path, img_name + ".jpg"))
            img = img/256


        X[i,:,:,:] = img
        y[i,:] = t_label[i]


    for i in range(val_nr):
        #print(i)
        img_name = val_img[i]
        if img_name[len(img_name)-2]=='.':
          channel_shift = img_name[len(img_name)-1]
          img_name = img_name[:-2]
          img = imread(os.path.join(path, img_name + ".jpg"))
          img = augment(img,channel_shift)
        else:
          img = imread(os.path.join(path, img_name + ".jpg"))

        img = img/256

        X_val[i,:,:,:] = img
        y_val[i,:] = val_label[i]

    return X, y, X_val, y_val


def testdataloader():

    img_list = []
    label_list = []
    path = 'data/Test_dataset'

    with open('data/Test_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            img_list.append(row[0])
            label = np.zeros(4)
            for i in range(1,5):
                if row[i] == '1.0':
                    label[i-1] = 1
                    break
            label_list.append(label)

    data = list(zip(img_list, label_list))
    shuffle(data)
    img_list[:], label_list[:] = zip(*data)

    total = len(img_list)

    X = np.zeros((total, 256, 256, 3), dtype=np.float32)
    y = np.zeros((total, 4), dtype=np.float32)

    for i in range(total):
        #print(i)
        img = imread(os.path.join(path, img_list[i] + ".jpg"))
        img = img/256

        X[i,:,:,:] = img
        y[i,:] = label_list[i]

    return X, y
