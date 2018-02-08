import cv2
import numpy as np
from torchvision import models

import utils
import utils2

if __name__ == '__main__':
    grad_cam = utils.GradCam(model=models.vgg19(pretrained=True), target_layer=35, target_category=None)

    img = cv2.imread('both.png', 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = utils2.preprocess_image(img)

    mask = grad_cam(input)

    utils2.show_cam_on_image(img, mask)
