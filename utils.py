import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            crop_size (int): Determines what size to crop of video. Defaults to 112.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, crop_size=112):
        self.original_dir = os.path.join('data', dataset)
        self.preprocessed_dir = os.path.join('data', 'preprocessed_' + dataset)
        self.split = split
        self.clip_len = clip_len
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = crop_size

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You need to download it from official website.')

        if not self.check_preprocess():
            print('Preprocessing {} split of {} dataset, this will take long, '
                  'but it will be done only once.'.format(self.split, dataset))
            self.preprocess()

        self.file_names, labels = [], []
        for label in os.listdir(os.path.join(self.preprocessed_dir, self.split)):
            for file_name in os.listdir(os.path.join(self.preprocessed_dir, self.split, label)):
                self.file_names.append(os.path.join(self.preprocessed_dir, self.split, label, file_name))
                labels.append(label)

        print('Number of {} videos: {:d}'.format(split, len(self.file_names)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Load and preprocess.
        buffer = self.load_frames(self.file_names[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        label = np.array(self.label_array[index])

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

        for file in os.listdir(os.path.join(self.original_dir, self.split)):
            os.mkdir(os.path.join(self.preprocessed_dir, self.split, file))

            for video in os.listdir(os.path.join(self.original_dir, self.split, file)):
                video_name = os.path.join(self.original_dir, self.split, file, video)
                save_name = os.path.join(self.preprocessed_dir, self.split, file, video)
                self.process_video(video_name, save_name)

        print('Preprocessing finished.')

    def process_video(self, video_name, save_name):
        # Initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(video_name)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure the preprocessed video has at least 16 frames
        extract_frequency = 4
        if frame_count // extract_frequency <= 16:
            extract_frequency -= 1
            if frame_count // extract_frequency <= 16:
                extract_frequency -= 1
                if frame_count // extract_frequency <= 16:
                    extract_frequency -= 1

        count = 0
        i = 0
        retaining = True

        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % extract_frequency == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_name.split('.')[0], '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # Randomly select time index for temporal jitter
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


def load_data(batch_size=64):
    train_data = VideoDataset(dataset='ucf101', split='train', clip_len=8)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = VideoDataset(dataset='ucf101', split='val', clip_len=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_data = VideoDataset(dataset='ucf101', split='test', clip_len=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()
