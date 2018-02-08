import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models

import utils


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img = transforms.Normalize(means, stds)(img)
    input = Variable(img.unsqueeze(0))
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img / 255)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    grad_cam = utils.GradCam(model=models.vgg19(pretrained=True), target_layer=35, target_category=None)

    img = cv2.imread('both.png', 1)
    input = preprocess_image(transforms.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    mask = grad_cam(input)

    show_cam_on_image(img, mask)
