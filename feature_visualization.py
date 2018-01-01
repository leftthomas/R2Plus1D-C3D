import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms


class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def __call__(self, x, index=None):
        if torch.cuda.is_available():
            features, output = self.extractor(x.cuda())
        else:
            features, output = self.extractor(x)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros(output.size())
        one_hot[0][index] = 1
        one_hot = Variable(one_hot)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for j, w in enumerate(weights):
            cam += w * target[j, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, IMAGE_SIZE)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Visualization')
    parser.add_argument('--image_path', type=str, help='input image path')
    parser.add_argument('--target_index', default=None, help='the index of scoring category')
    opt = parser.parse_args()
    IMAGE_PATH = opt.image_path
    TARGET_INDEX = opt.target_index

    image = Image.open(IMAGE_PATH)
    IMAGE_SIZE = image.size
    image = transforms.Resize(IMAGE_SIZE)(image)
    image = transforms.ToTensor()(image)
    image = Variable(image.unsqueeze(dim=0))

    net = models.vgg19(pretrained=True)
    grad_cam = GradCam(model=net, target_layer_names=["35"])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    show_cam_on_image(image.squeeze(dim=0).data.numpy().transpose(1, 2, 0), grad_cam(image, TARGET_INDEX))
