import argparse

from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

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
