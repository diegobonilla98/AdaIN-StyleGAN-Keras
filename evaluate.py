import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from AdaIN_ST_Functions import StyleTransfer
from BoniDL import utils
from tensorflow.keras import backend as K

utils.allow_gpu_growth()

weights_path = './results/weights/epoch_35300.h5'
st = StyleTransfer(trained_weights=weights_path)
model = st.full_model
model.summary()


def adjust_image(image, gamma=1.0, alpha=1.5, beta=0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


norm = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255.
output = model.predict([st.test_contents, st.test_styles])

contents = np.hstack([norm(st.test_contents[i]) for i in range(len(st.test_styles))])
styles = np.hstack([norm(st.test_styles[i]) for i in range(len(st.test_styles))])
outputs = np.hstack([adjust_image(norm(output[i]).astype('uint8')) for i in range(len(st.test_styles))])
mix = np.vstack([contents, styles, outputs]).astype('uint8')

cv2.imwrite('result.png', mix)
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.imshow('Output', mix)
cv2.waitKey()
