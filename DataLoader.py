import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2
from functools import reduce
import operator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import vgg19


class DataLoader(Sequence):
    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.size = (size, size)
        styles = ['Abstract_Expressionism', 'Impressionism', 'Synthetic_Cubism', 'Color_Field_Painting', 'Analytical_Cubism', 'Action_painting', 'Expressionism', 'Pointillism', 'Symbolism', 'Post_Impressionism', 'Cubism', 'Minimalism']
        self.STYLES = [glob.glob(os.path.join('/media/bonilla/My Book/wikiart', p, '*')) for p in styles]
        self.STYLES = reduce(operator.concat, self.STYLES)
        self.CONTENTS = glob.glob('/media/bonilla/My Book/coco/train2017/*')
        self.n = len(self.CONTENTS)

    def _load_image(self, image_path):
        image = load_img(image_path, target_size=self.size)
        image = img_to_array(image)
        image = image[np.newaxis, :, :, :]
        image = vgg19.preprocess_input(image)[0]
        # image = image[:, :, ::-1].astype('float32') / 255.
        return image

    def __getitem__(self, index):
        images = []
        styles = []
        images_path = np.random.choice(self.CONTENTS, self.batch_size)
        styles_path = np.random.choice(self.STYLES, self.batch_size)
        for image_path, style_path in zip(images_path, styles_path):
            image = self._load_image(image_path)
            style = self._load_image(style_path)
            images.append(image)
            styles.append(style)
        images = np.array(images)
        styles = np.array(styles)
        return images, styles

    def __len__(self):
        return self.n // self.batch_size


if __name__ == '__main__':
    dl = DataLoader(512, 4)
    d = dl[0]
