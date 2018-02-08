import cv2
import torchvision.transforms as transforms
from torchvision import models

import utils
import utils2

if __name__ == '__main__':
    grad_cam = utils.GradCam(model=models.vgg19(pretrained=True), target_layer=35, target_category=None)

    img = cv2.imread('both.png', 1)
    input = utils2.preprocess_image(transforms.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    mask = grad_cam(input)

    utils2.show_cam_on_image(img, mask)
