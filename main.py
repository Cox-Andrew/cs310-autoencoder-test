from os import listdir, getenv
from os.path import isfile, join, isdir
from PIL import Image
import numpy as np
import shelve
from tensorflow import keras
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, \
    LayerNormalization
from tensorflow.keras.models import Sequential, load_model


# import matplotlib.pyplot as plt


class Config:
    DATASET_ROOT = join(getenv("OneDrive"), r"project\literature\data-sets\UCSD_Anomaly_Dataset.v1p2\UCSDped1")
    DATASET_PATH = join(DATASET_ROOT, r'Train')
    SINGLE_TEST_PATH = join(DATASET_ROOT, r'Test', r'Test032')
    BATCH_SIZE = 1
    EPOCHS = 1
    MODEL_PATH = r'C:\Users\valsp\source\repos\cs310-autoencoder-test\model_lstm.hdf5'


def get_clips_by_stride(stride, frames_list, sequence_size):
    """
    stride: The desired distance between two consecutive frames
    frames_list: list of sorted frames 256 X 256
    sequence_size: size of LSTM sequence
    returns A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips


def get_training_set():
    """
    list: A list of training sequences (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    #####################################
    # cache = shelve.open(Config.CACHE_PATH)
    # return cache["datasetLSTM"]
    #####################################
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in sorted(listdir(Config.DATASET_PATH)):
        if isdir(join(Config.DATASET_PATH, f)):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(join(Config.DATASET_PATH, f))):
                if str(join(join(Config.DATASET_PATH, f), c))[-3:] == "tif":
                    # TODO: CURRENTLY GET SOME WEIRD ERROR HERE exit code -1073740791 (0xC0000409)
                    img = Image.open(join(join(Config.DATASET_PATH, f), c)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips


def get_model(reload_model=True):
    if not reload_model:
        return load_model(Config.MODEL_PATH, custom_objects={'LayerNormalization': LayerNormalization})

    training_set = get_training_set()
    training_set = np.array(training_set)
    training_set = training_set.reshape(-1, 10, 256, 256, 1)

    seq = Sequential()

    seq.add(
        TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))

    print(seq.summary())

    seq.compile(loss='mse', optimizer=keras.optimisers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    seq.fit(training_set, training_set,
            batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)

    seq.save(Config.MODEL_PATH)

    return seq


def get_single_test():
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "tif":
            img = Image.open(join(Config.SINGLE_TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test


def evaluate():
    model = get_model(reload_model=True)
    print("trained model")
    test = get_single_test()
    print(test.shape)
    sz = test.shape[0] - 10 + 1
    sequences = np.zeros((sz, 10, 256, 256, 1))

    # sliding window
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    print("got data")

    # get the reconstruction cost
    reconstructed_sequences = model.predict(sequences, batch_size=4)
    sequences_reconstruction_cost = np.array(
        [np.linalg.norm(np.subtract(sequences[i], reconstructed_sequences[i])) for i in range(0, sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    # plt.plot(sr)
    # plt.ylabel('regularity score Sr(t)')
    # plt.xlabel('frame t')
    # plt.show()


if __name__ == '__main__':
    evaluate()
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
