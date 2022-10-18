"""
DATASET.py can divide the COCO dataset into training set and validation set in a desired proportion
'image_path' = 'Absolute position of images in the dataset'
'label_path' = 'Absolute position of labels in the dataset'
'new_file_path' = 'Location of the output file'
train_rate is the percentage of the training set
val_rate is the percentage of the validation set
"""
import os
import shutil
import random

random.seed(0)


def split_data(image_path, new_file_path, train_rate, val_rate):
    eachclass_image = []
    for image in os.listdir(image_path):
        eachclass_image.append(image)
    total = len(eachclass_image)
    random.shuffle(eachclass_image)
    train_images = eachclass_image[0:int(train_rate * total)]
    val_images = eachclass_image[int(train_rate * total):int((train_rate + val_rate) * total)]

    # training set
    for image in train_images:
        print(image)
        old_path = image_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        # print(new_path)
        shutil.copy(old_path, new_path)
    new_name = os.listdir(new_file_path + '/' + 'train' + '/' + 'images')
    # print(new_name[1][:-4])
    for im in new_name:
        old_xmlpath = label_path + '/' + im[:-3] + 'txt'
        print('old', old_xmlpath)
        new_xmlpath1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_xmlpath1):
            os.makedirs(new_xmlpath1)
        new_xmlpath = new_xmlpath1 + '/' + im[:-3] + 'txt'
        print('xml name', new_xmlpath)
        if not os.path.exists(f'{old_xmlpath}'):
            open(f'{old_xmlpath}', 'w')
        shutil.copy(old_xmlpath, new_xmlpath)

    # validation set
    for image in val_images:
        old_path = image_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
    new_name = os.listdir(new_file_path + '/' + 'val' + '/' + 'images')
    for im in new_name:
        old_xmlpath = label_path + '/' + im[:-3] + 'txt'
        new_xmlpath1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_xmlpath1):
            os.makedirs(new_xmlpath1)
        new_xmlpath = new_xmlpath1 + '/' + im[:-3] + 'txt'
        if not os.path.exists(f'{old_xmlpath}'):
            open(f'{old_xmlpath}', 'w')
        shutil.copy(old_xmlpath, new_xmlpath)
    print('ok')


if __name__ == '__main__':
    image_path = ''
    label_path = ''
    new_file_path = ''
    split_data(image_path, new_file_path, train_rate=0.7, val_rate=0.3)
