import os
import cv2
import imageio
import numpy as np
import pandas as pd

from tensorflow.keras import layers
from tensorflow import keras

from source.config import MAX_SEQ_LENGTH, NUM_FEATURES, IMG_SIZE


center_crop_layer = layers.experimental.preprocessing.CenterCrop(IMG_SIZE, IMG_SIZE)


def read_datasets(classnames=False):
    train_df = pd.read_csv("source/data/train.tsv", sep="\t")
    class_vocab = train_df.iloc[:, 1:].columns.tolist()
    if classnames:
        return class_vocab
    else:
        return train_df


def structure(data):
    data.set_index("File Name", inplace=True)
    data = data.applymap(str.lower)
    data.columns = [col.lower() for col in data.columns.tolist()]
    data_one_hot_encoded = pd.get_dummies(data).reset_index()
    data_one_hot_encoded.to_csv("source/data/train.tsv", sep="\t", index=False)


def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped


# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center(frame)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["File Name"].values.tolist()
    labels = df.iloc[:, 1:].values
    # labels = label_processor(labels).numpy()

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))

        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate((frames, padding))

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_featutes = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[1]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])
                else:
                    temp_frame_featutes[i, j, :] = 0.0

        frame_features[idx,] = temp_frame_featutes.squeeze()

    return frame_features, labels


def prepare_single_video(frames):
    frame_featutes = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        try:
            frames = np.concatenate((frames, padding))
        except ValueError:
            pass
            # frames = np.concatenate((frames[:, None, None, None], padding))

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        try:
            video_length = batch.shape[1]
        except IndexError:
            continue
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_featutes[i, j, :] = 0.0

    return frame_featutes


def predict_action(model, path, verbose=False, return_frames=False):
    class_vocab = read_datasets(classnames=True)
    frames = load_video(os.path.join(path))
    frame_features = prepare_single_video(frames)
    probabilities = model.predict(frame_features)

    if verbose:
        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

    if return_frames:
        return frames

    df = pd.DataFrame(probabilities, columns=class_vocab).T
    df['group'] = df.index.str.split("_").str[0]
    gestures = df.groupby("group")[0].idxmax().tolist()
    d = {}
    for gesture in gestures:
        d[gesture.split("_")[0].title()] = gesture.split("_")[1].title()

    if d['Gesture'] == "No":
        for k in d.keys():
            if k != 'Gesture':
                d[k] = "No Gesture"
    return d
    # return ["{}: {}".format(gesture.split("_")[0].title(), gesture.split("_")[1].title()) for gesture in gestures]


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
