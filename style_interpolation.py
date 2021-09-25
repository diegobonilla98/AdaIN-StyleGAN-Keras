import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import AdaIN_ST_Functions
from BoniDL import utils
from tensorflow.keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

weights_path = './results/weights/epoch_35300.h5'
st = AdaIN_ST_Functions.StyleTransfer(trained_weights=weights_path)
f_model = st.first_feat_extractor
decoder = st.decoder


def adjust_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


norm = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255.

outputs = []
n_inter_images = 5
img_idx = 7
content = np.expand_dims(st.test_contents[img_idx], axis=0)
style = np.expand_dims(st.test_styles[img_idx], axis=0)
for alpha in np.linspace(0., 1., n_inter_images, dtype='float32'):
    content_feats = f_model(content)
    style_feats = f_model(style)
    content_weighted = (1. - alpha) * content_feats
    combined_weighted = alpha * st.AdaIN(inputs=[content_feats, style_feats])
    mix = decoder(content_weighted + combined_weighted).eval(session=session)[0]
    outputs.append(mix)

outputs_comb = np.concatenate([adjust_image(norm(outputs[i]).astype('uint8'))[np.newaxis, :, :, :] for i in range(len(outputs))], axis=0)
outputs_comb = np.concatenate([norm(content).astype('uint8'), outputs_comb], axis=0)
outputs_comb = np.concatenate([outputs_comb, norm(style).astype('uint8')], axis=0)
mix = np.hstack([outputs_comb[i] for i in range(len(outputs_comb))])

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.imshow('Output', mix)
key = cv2.waitKey()
if key == ord(' '):
    name = input("Type image name: ")
    cv2.imwrite(f'{name}.png', mix)
