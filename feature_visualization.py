import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms


class GradCam:
    def __init__(self, model, target_layer_names, target_index):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = model.cuda()
        self.target_layers = target_layer_names
        self.target_index = target_index
        self.features = []
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        # save the target layers' gradients and features, then get the classes scores
        if torch.cuda.is_available():
            x = x.cuda()
        for name, module in self.model.features.named_children():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                self.features += [x]
        x = x.view(x.size(0), -1)
        output = self.model.classifier(x)

        if self.target_index is None:
            self.target_index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros(output.size())
        one_hot[0][self.target_index] = 1
        one_hot = Variable(one_hot)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients[-1].cpu().data.numpy()

        target = self.features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for j, w in enumerate(weights):
            cam += w * target[j, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Visualization')
    parser.add_argument('--image_path', type=str, help='input image path')
    parser.add_argument('--target_index', default=None, help='the index of scoring category')
    opt = parser.parse_args()
    IMAGE_PATH = opt.image_path
    TARGET_INDEX = opt.target_index

    image = Image.open(IMAGE_PATH)
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    image = Variable(image.unsqueeze(dim=0))

    net = models.vgg19(pretrained=True)
    grad_cam = GradCam(net, ['35'], TARGET_INDEX)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    show_cam_on_image(image.squeeze(dim=0).data.numpy().transpose(1, 2, 0), grad_cam(image))
