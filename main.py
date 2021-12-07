from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import model as modelbuilder
import os
import numpy as np

import matplotlib.pyplot as plt
import random

TRAIN_PATH='D:/data-science-bowl/stage1_train/'
TEST_PATH='D:/data-science-bowl/stage1_test/'

IMG_WIDTH = 128
IMG_HEIGTH = 128
IMG_CHANNELS = 3
seed =42

np.random.seed(seed)
np.random.seed(seed)

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGTH, IMG_WIDTH, 1), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGTH, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  # Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGTH, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGTH, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGTH, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done')

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


model = modelbuilder.build_model(IMG_WIDTH, IMG_HEIGTH, IMG_CHANNELS)
callback = modelbuilder.define_callback()

results = model.fit(X_train, Y_train, validation_steps=0.1, callbacks=callback, batch_size=16, epochs=25)



