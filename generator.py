import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
import random
import PIL


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.page = 1
        self.epoch = 0
        self.list_dir = os.listdir(self.file_path)
        if shuffle:
            np.random.shuffle(self.list_dir)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # return images, labels

        start_index = self.page * self.batch_size - self.batch_size
        end_index = self.page * self.batch_size
        self.page += 1

        images = []
        for image in self.list_dir[start_index: end_index]:
            image_file = self.augment(np.load(f'{self.file_path}/{image}'))
            images.append({'name': image, 'data': resize(image_file, tuple(self.image_size))})

        if len(images) < self.batch_size:
            if self.shuffle:
                np.random.shuffle(self.list_dir)
                self.page = 1

            count = self.batch_size - len(images)
            self.epoch += 1
            for i in range(0, count):
                image_file = self.augment(np.load(f'{self.file_path}{self.list_dir[i]}'))
                images.append({'name': self.list_dir[i], 'data': resize(image_file, tuple(self.image_size))})
        images_numbers = [image['name'][:len(image['name']) - 4] for image in images]

        with open(self.label_path) as json_file:
            data = json.load(json_file)
            labels = [data[i] for i in images_numbers]

        return np.array([image['data'] for image in images]), labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.rotation:
            rotation_degree = np.random.choice([90, 180, 270], 1)
            img = rotate(img, rotation_degree[0])
        if self.mirroring:
            img = np.flip(img)

        return img

    def current_epoch(self):
        # return the current epoch number

        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input

        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()

        fig = plt.figure(1)

        for k in range(0, len(images)):
            label = self.class_name(labels[k])
            ax = fig.add_subplot(self.batch_size // 3, 3, k + 1, frameon=False, title=label, ymargin=10)
            ax.imshow(images[k])

            ax.axis('off')
        plt.tight_layout()
        plt.show()
