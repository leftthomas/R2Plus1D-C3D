import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision import models

import utils


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) / 255)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    grad_cam = utils.GradCam(model=models.vgg19(pretrained=True), target_layer=35, target_category=None)

    img = Image.open('both.png')
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    input = Variable(transform_test(img).unsqueeze(0))

    mask = grad_cam(input)

    show_cam_on_image(np.array(img), mask)
