from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from scipy.spatial import distance as dist
import cv2
from numpy.random import seed
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import glob
import json
import label_validation_func as lf
import pytest

seed(200)
tf.compat.v1.random.set_random_seed(1000)


def test_generate_linear_model():

    l2_reg = 0
    label_images_dir = "."

    ConfigJson = {}
    with open("fraud_label_config.json", "r") as inp_file:
        config_json = json.load(inp_file)
    model_name = config_json["model_name"]
    image_path_name = label_images_dir + "/" + config_json["target_directory"] + "/"
    percent_thres = float(config_json["scale_factor_for_threshold"])
    target_image_file_name = image_path_name + config_json["target_image_file_name"]

    model_1 = Sequential()
    img_shape = (224, 224, 3)
    model_1.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=(224, 224, 3),
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
    )
    model_1.add(BatchNormalization())
    model_1.add(Activation("relu"))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 2
    model_1.add(Conv2D(256, (5, 5), padding="same"))
    model_1.add(BatchNormalization())
    model_1.add(Activation("relu"))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    x = np.zeros((2, 224, 224, 3))
    l2_reg = 0

    [model_label, be_l, num_bins, max_histor_level] = lf.generate_model_linear(
        target_image_file_name, l2_reg, x
    )

    test_output = [model_1.output_shape]
    return_list = [model_label.output_shape]

    assert return_list == test_output


def test_generate_art_model():

    label_images_dir = "."

    ConfigJson = {}
    with open("fraud_label_config.json", "r") as inp_file:
        config_json = json.load(inp_file)
    model_name = config_json["model_name"]
    image_path_name = label_images_dir + "/" + config_json["target_directory"] + "/"
    percent_thres = float(config_json["scale_factor_for_threshold"])
    target_image_file_name = image_path_name + config_json["target_image_file_name"]
    l2_reg = 0
    label_model = Sequential()
    img_shape = (224, 224, 3)
    label_model.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=(224, 224, 3),
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
    )
    label_model.add(BatchNormalization())
    label_model.add(Activation("relu"))
    label_model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 2
    label_model.add(Conv2D(256, (5, 5), padding="same"))
    label_model.add(BatchNormalization())
    label_model.add(Activation("relu"))
    label_model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 3
    label_model.add(ZeroPadding2D((1, 1)))
    label_model.add(Conv2D(512, (3, 3), padding="same"))
    label_model.add(BatchNormalization())
    label_model.add(Activation("relu"))
    label_model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 4
    label_model.add(ZeroPadding2D((1, 1)))
    label_model.add(Conv2D(1024, (3, 3), padding="same"))
    label_model.add(BatchNormalization())
    label_model.add(Activation("relu"))
    # Layer 5
    label_model.add(ZeroPadding2D((1, 1)))
    label_model.add(Conv2D(1024, (3, 3), padding="same"))
    label_model.add(BatchNormalization())
    label_model.add(Activation("relu"))
    label_model.add(MaxPooling2D(pool_size=(2, 2)))

    l2_reg = 0
    x = np.zeros((2, 224, 224, 3))
    [lb_model, similar_diff, x_only] = lf.generate_model_art(
        target_image_file_name, l2_reg, x
    )
    return_list = [lb_model.output_shape, similar_diff]
    s_diff = 12.4 / 0.6
    test_output = [label_model.output_shape, s_diff]

    assert return_list == test_output


def test_label_art_model():
    label_images_dir = "."

    seed(200)
    tf.compat.v1.random.set_random_seed(1000)

    ConfigJson = {}
    with open("fraud_label_config.json", "r") as inp_file:
        config_json = json.load(inp_file)
    model_name = config_json["model_name"]
    image_path_name = label_images_dir + "/" + config_json["target_directory"] + "/"
    percent_thres = float(config_json["scale_factor_for_threshold"])
    target_image_file_name = image_path_name + config_json["target_image_file_name"]
    x = np.zeros((2, 224, 224, 3))

    l2_reg = 0
    mypath = image_path_name + "/"

    [lb_model, similar_diff, x_only] = lf.generate_model_art(
        target_image_file_name, l2_reg, x
    )
    file_name_list = lf.display_fraudulent_labels_art(
        mypath, percent_thres, lb_model, similar_diff, target_image_file_name, x_only, x
    )

    test_output = [
        "./test_data/Beer_Label_22.jpg",
        "./test_data/Bud_Label_LessColors.jpg",
        "./test_data/Bud_Label_hozFlip.jpg",
        "./test_data/Beer_Label_44.jpg",
    ]
    assert file_name_list == test_output


def test_label_linear_model():

    label_images_dir = "."

    seed(200)
    tf.compat.v1.random.set_random_seed(1000)

    config_json = {}
    with open("fraud_label_config.json", "r") as inp_file:
        config_json = json.load(inp_file)
    model_name = config_json["model_name"]
    image_path_name = label_images_dir + "/" + config_json["target_directory"] + "/"
    percent_thres = float(config_json["scale_factor_for_threshold"])
    target_image_file_name = image_path_name + config_json["target_image_file_name"]
    x = np.zeros((2, 224, 224, 3))

    l2_reg = 0
    mypath = image_path_name + "/"

    [lb_model, be_l, num_bins, max_histo_level] = lf.generate_model_linear(
        target_image_file_name, l2_reg, x
    )
    file_name_list = lf.display_fraudulent_labels_linear(
        mypath,
        percent_thres,
        lb_model,
        be_l,
        num_bins,
        max_histo_level,
        target_image_file_name,
        x,
    )

    test_output = [
        "./test_data/Bud_Label_hozFlip.jpg",
        "./test_data/Beer_Label_44.jpg",
    ]

    assert file_name_list == test_output
