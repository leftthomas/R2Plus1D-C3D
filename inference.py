import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils
from models.C3D import C3D
from models.R2Plus1D import R2Plus1D


def center_crop(image):
    height_index = math.floor((image.shape[0] - crop_size) / 2)
    width_index = math.floor((image.shape[1] - crop_size) / 2)
    image = image[height_index:height_index + crop_size, width_index:width_index + crop_size, :]
    return np.array(image).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Activity Recognition')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics600'],
                        help='dataset type')
    parser.add_argument('--model_type', default='r2plus1d', type=str, choices=['r2plus1d', 'c3d'], help='model type')
    parser.add_argument('--video_name', type=str, help='test video name')
    parser.add_argument('--model_name', default='ucf101_r2plus1d.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE, MODEL_TYPE, VIDEO_NAME, MODEL_NAME = opt.data_type, opt.model_type, opt.video_name, opt.model_name

    clip_len, resize_height, crop_size, = utils.CLIP_LEN, utils.RESIZE_HEIGHT, utils.CROP_SIZE
    class_names = utils.get_labels(DATA_TYPE)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if '{}_{}.pth'.format(DATA_TYPE, MODEL_TYPE) != MODEL_NAME:
        raise NotImplementedError('the model name must be the same model type and same data type')

    if MODEL_TYPE == 'r2plus1d':
        model = R2Plus1D(len(class_names), (2, 2, 2, 2))
    else:
        model = C3D(len(class_names))

    checkpoint = torch.load('epochs/{}'.format(MODEL_NAME), map_location=lambda storage, loc: storage)
    model = model.load_state_dict(checkpoint).to(DEVICE).eval()

    # read video
    cap, retaining, clips = cv2.VideoCapture(VIDEO_NAME), True, []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        resize_width = math.floor(frame.shape[1] / frame.shape[0] * resize_height)
        # make sure it can be cropped correctly
        if resize_width < crop_size:
            resize_width = resize_height
            resize_height = math.floor(frame.shape[0] / frame.shape[1] * resize_width)
        tmp_ = center_crop(cv2.resize(frame, (resize_width, resize_height)))
        tmp = tmp_.astype(np.float32) / 255.0
        clips.append(tmp)
        if len(clips) == clip_len or len(clips) == frame_count:
            inputs = np.array(clips)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs).to(DEVICE)
            with torch.no_grad():
                outputs = model.forward(inputs)

            prob = F.softmax(dim=-1)(outputs)
            label = torch.max(prob, -1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, 'prob: %.4f' % prob[0][label], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            clips.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
