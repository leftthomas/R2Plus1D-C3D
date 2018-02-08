import cv2
import numpy as np
import torch
from torch.autograd import Variable


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer, target_category):
        self.model = model.eval()
        self.target_layer = len(model.features) - 1 if target_layer is None else target_layer
        if self.target_layer > len(model.features) - 1:
            raise ValueError(
                "Expected target layer must less than the total layers({}) of features.".format(len(model.features)))
        self.target_category = target_category
        self.features = None
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        x = input
        if torch.cuda.is_available():
            x = input.cuda()

        for idx, module in enumerate(self.model.features.children()):
            x = module(x)
            if idx == self.target_layer:
                x.register_hook(self.save_gradient)
                self.features = x
        output = x.view(x.size(0), -1)
        output = self.model.classifier(output)

        if self.target_category is None:
            self.target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][self.target_category] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if torch.cuda.is_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients.cpu().data.numpy()
        target = self.features.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
