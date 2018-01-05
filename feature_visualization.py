import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms


class GradCam:
    def __init__(self, model, target_layer, target_category):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = model.cuda()
        self.target_layer = target_layer
        self.target_category = target_category
        self.features = None
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        # save the target layer' gradients and features, then get the category scores
        if torch.cuda.is_available():
            x = x.cuda()
        for name, module in self.model.features.named_children():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                self.features = x
        x = x.view(x.size(0), -1)
        output = self.model.classifier(x)

        # if the target category equal None, return the feature map of the highest scoring category,
        # otherwise, return the feature map of the requested category
        if self.target_category is None:
            one_hot, self.target_category = output.max(dim=-1)
        else:
            one_hot = output[0][self.target_category]
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward()

        weights = self.gradients.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        cam = F.relu((weights * self.features).sum(dim=1))
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = transforms.ToPILImage()(cam.data.cpu())
        cam = transforms.Resize(size=(224, 224))(cam)
        return cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Visualization')
    parser.add_argument('--image_path', type=str, help='input image path')
    parser.add_argument('--target_category', default=None, help='the category of visualization')
    opt = parser.parse_args()
    IMAGE_PATH = opt.image_path
    TARGET_CATEGORY = opt.target_category

    img = Image.open(IMAGE_PATH)
    img = transforms.Resize((224, 224))(img)
    image = transforms.ToTensor()(img)
    image = Variable(image.unsqueeze(dim=0))

    net = models.vgg19(pretrained=True)
    grad_cam = GradCam(net, str(35), TARGET_CATEGORY)
    mask = grad_cam(image)
    result = Image.new('RGBA', img.size)
    result.paste(im=img, mask=mask)
    result.save('cam.png')
