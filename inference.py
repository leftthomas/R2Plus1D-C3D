import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import Network


def center_crop(image):
    image = image[8:120, 30:142, :]
    return np.array(image).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Activity Recognition')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51'], help='dataset type')
    parser.add_argument('--clip_len', default=16, type=int, help='number of frames in each video')
    parser.add_argument('--crop_size', default=112, type=int, help='crop size of video')
    parser.add_argument('--video_name', type=str, help='test video name')
    parser.add_argument('--model_name', default='ucf101_100.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    CLIP_LEN = opt.clip_len
    CROP_SIZE = opt.crop_size
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name

    with open('data/{}_labels.txt'.format(DATA_TYPE), 'r') as f:
        class_names = f.readlines()
        f.close()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if DATA_TYPE == 'ucf101':
        NUM_CLASS = 101
    else:
        NUM_CLASS = 51

    model = Network(NUM_CLASS)
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
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
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
