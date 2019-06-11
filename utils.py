import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# global configs
CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE = 32, 128, 112


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
    """

    def __init__(self, dataset='ucf101', split='train'):
        self.original_dir = os.path.join('data', dataset)
        self.preprocessed_dir = os.path.join('data', 'preprocessed_' + dataset)
        self.split = split

        if not self.check_integrity():
            raise RuntimeError('{} split of {} dataset is not found. You need to '
                               'download it from official website.'.format(split, dataset))

        if not self.check_preprocess():
            print('Preprocessing {} split of {} dataset, this will take long, '
                  'but it will be done only once.'.format(split, dataset))
            self.preprocess()

        self.file_names, labels = [], []
        for label in sorted(os.listdir(os.path.join(self.preprocessed_dir, self.split))):
            for file_name in sorted(os.listdir(os.path.join(self.preprocessed_dir, self.split, label))):
                self.file_names.append(os.path.join(self.preprocessed_dir, self.split, label, file_name))
                labels.append(label)

        print('Number of {} videos: {:d}'.format(split, len(self.file_names)))

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(get_labels(dataset))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # load and preprocess.
        buffer = self.load_frames(self.file_names[index])
        buffer = self.crop(buffer, CLIP_LEN, CROP_SIZE)
        label = np.array(self.label_array[index])
        if self.split == 'train':
            # perform data augmentation (random horizontal flip)
            buffer = self.random_flip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(label)

    def check_integrity(self):
        if os.path.exists(os.path.join(self.original_dir, self.split)):
            return True
        else:
            return False

    def check_preprocess(self):
        if os.path.exists(os.path.join(self.preprocessed_dir, self.split)):
            return True
        else:
            return False

    def preprocess(self):
        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)
        os.mkdir(os.path.join(self.preprocessed_dir, self.split))

        for file in sorted(os.listdir(os.path.join(self.original_dir, self.split))):
            os.mkdir(os.path.join(self.preprocessed_dir, self.split, file))

            for video in sorted(os.listdir(os.path.join(self.original_dir, self.split, file))):
                video_name = os.path.join(self.original_dir, self.split, file, video)
                save_name = os.path.join(self.preprocessed_dir, self.split, file, video)
                self.process_video(video_name, save_name)

        print('Preprocess finished.')

    @staticmethod
    def process_video(video_name, save_name):
        print('Preprocess {}'.format(video_name))
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(video_name)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # make sure the preprocessed video has at least CLIP_LEN frames
        extract_frequency = 4
        if frame_count // extract_frequency <= CLIP_LEN:
            extract_frequency -= 1
            if frame_count // extract_frequency <= CLIP_LEN:
                extract_frequency -= 1
                if frame_count // extract_frequency <= CLIP_LEN:
                    extract_frequency -= 1

        count, i, retaining = 0, 0, True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % extract_frequency == 0:
                resize_height = RESIZE_HEIGHT
                resize_width = math.floor(frame_width / frame_height * resize_height)
                # make sure resize width >= crop size
                if resize_width < CROP_SIZE:
                    resize_width = RESIZE_HEIGHT
                    resize_height = math.floor(frame_height / frame_width * resize_width)

                frame = cv2.resize(frame, (resize_width, resize_height))
                if not os.path.exists(save_name.split('.')[0]):
                    os.mkdir(save_name.split('.')[0])
                cv2.imwrite(filename=os.path.join(save_name.split('.')[0], '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

    @staticmethod
    def random_flip(buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = frame

        return buffer

    @staticmethod
    def normalize(buffer):
        buffer = buffer.astype(np.float32)
        for i, frame in enumerate(buffer):
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    @staticmethod
    def to_tensor(buffer):
        return buffer.transpose((3, 0, 1, 2))

    @staticmethod
    def load_frames(file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = []
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name))
            buffer.append(frame)

        return np.array(buffer).astype(np.uint8)

    def crop(self, buffer, clip_len, crop_size):
        if self.split == 'train':
            # randomly select time index for temporal jitter
            if buffer.shape[0] > clip_len:
                time_index = np.random.randint(buffer.shape[0] - clip_len)
            else:
                time_index = 0
            # randomly select start indices in order to crop the video
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            # crop and jitter the video using indexing. The spatial crop is performed on
            # the entire array, so each frame is cropped in the same location. The temporal
            # jitter takes place via the selection of consecutive frames
        else:
            # for val and test, select the middle and center frames
            if buffer.shape[0] > clip_len:
                time_index = math.floor((buffer.shape[0] - clip_len) / 2)
            else:
                time_index = 0
            height_index = math.floor((buffer.shape[1] - crop_size) / 2)
            width_index = math.floor((buffer.shape[2] - crop_size) / 2)
        buffer = buffer[time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer


def load_data(dataset='ucf101', batch_size=8):
    train_data = VideoDataset(dataset=dataset, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_data = VideoDataset(dataset=dataset, split='val')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_data = VideoDataset(dataset=dataset, split='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader, test_loader


def get_labels(dataset='ucf101'):
    labels = []
    with open('data/{}_labels.txt'.format(dataset), 'r') as load_f:
        raw_labels = load_f.readlines()
    for label in raw_labels:
        labels.append(label.replace('\n', ''))
    return sorted(labels)


