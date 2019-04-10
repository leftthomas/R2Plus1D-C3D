import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils
from model import Model


def center_crop(image):
    height_index = math.floor((image.shape[0] - 112) / 2)
    width_index = math.floor((image.shape[1] - 112) / 2)
    image = image[height_index:height_index + 112, width_index:width_index + 112, :]
    return np.array(image).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Activity Recognition')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics600'],
                        help='dataset type')
    parser.add_argument('--video_name', type=str, help='test video name')
    parser.add_argument('--model_name', default='ucf101.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name

    class_names = utils.get_labels(DATA_TYPE)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model(len(class_names))
    checkpoint = torch.load('epochs/{}'.format(MODEL_NAME), map_location=lambda storage, loc: storage)
    model = model.load_state_dict(checkpoint).to(DEVICE).eval()

    # read video
    cap = cv2.VideoCapture(VIDEO_NAME)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        resize_width = math.floor(frame.shape[1] / frame.shape[0] * 120)
        tmp_ = center_crop(cv2.resize(frame, (resize_width, 120)))
        tmp = tmp_.astype(np.float32) / 255.0
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs).to(DEVICE)
            with torch.no_grad():
                outputs = model.forward(inputs)

            prob = F.softmax(dim=-1)(outputs)
            label = torch.max(prob, -1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % prob[0][label], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
