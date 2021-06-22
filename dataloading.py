import tensorflow as tf
import librosa
import numpy as np
import tensorflow_io as tfio
import soundfile as sf
import os
import random
import IPython.display as ipd
import sklearn

import librosa as librosa
import multiprocessing as mp
from functools import partial
# print(os.environ)
config_filename = None
if "CLARITY_ROOT" in os.environ:
    config_filename = f"{os.environ['CLARITY_ROOT']}/clarity.cfg"
# PATH_TO_CLARITY_FOLDER = "/home/kenders/clarity_CEC1/"
PATH_TO_CLARITY_FOLDER = os.environ['CLARITY_ROOT']
import pathlib
import argparse
import json
from tqdm import tqdm
import sys
from scipy import signal

# install clarity core tools
from clarity_core.config import CONFIG

# sys.path.append(PATH_TO_CLARITY_FOLDER + r"../projects/GHA")
# from GHA import GHAHearingAid as HearingAid


def sync_y_to_x(x, y):
    """Synchronises y to x.
    Carries out a cross coreelation between x and y, uses the arg max of the
    result to determine the relative delay between the two. And applies the delay
    to y before returning the delayed version of y

    Args:
        x: The reference
        y: the signal that is to be synced
    Returns
        y_d: the delayed signal.

    """
    corr = signal.correlate(x, y, mode='full', method='auto')
    in2_len = x.shape[0]
    in1_len = y.shape[0]
    lags = np.arange(-in2_len + 1, in1_len)
    delay = lags[np.argmax(np.abs(corr))]
    if delay < 0:
        y = np.pad(y[delay:], [[0,delay] ]  )
    elif delay > 0:
        y = np.pad(y[0:-delay], [[delay,0] ]  )
    return y



def frame_audio(x, frame_size, frame_step):
    """Frame audio.

    Pre-pads `frame_size` zeros before splitting audio into `frame_size` frames, using frame_step
    to overlap frames. The pre-padding of `frame_size` zeros is so that we can extract the last
    `frame_step` samples of each frame and have a framed processing system that only looks ahead
    `frame_step` samples. Note as pre padding is `frame_size`, the first
    resulting frame is all because of padding and the signal actually starts
    with the second frame.

    Args:
        x: the audio (channels ,samples, )
        frame_size (int): The frame szie, the audio is prepended with zeros of this length in the
            samples dimension
        frame_step: The frame step
    Returns:
        x_framed: the framed audio (frames, frame_size, channels)

    """
    pad_n = frame_size
    x_padded = np.pad(x, [[0, 0], [pad_n, 0]])
    x_framed = tf.signal.frame(x_padded, frame_size, frame_step).numpy().astype(np.float32)
#     x_framed = librosa.util.frame(x_padded, frame_length, frame_step, axis=-1)

    return x_framed

def load_wav_at_sample_rate(filename, rate_out=16000):
    """ read in a waveform file and convert to rate_out mono """
    audio, sample_rate = librosa.load(filename, sr=rate_out, mono=False)
    # allways return 2 dim array:
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, 0)
    audio = np.transpose(audio, [1,0])  # channels last
    return audio.astype(np.float32)

def chunk_list(input_list, chunk_size):
    """Convert list into a list of lists."""
    return [
        input_list[chunk : chunk + chunk_size]
        for chunk in range(0, len(input_list), chunk_size)
    ]


def next_power_of_two(x):
    exponent = np.ceil(np.log2(x)/np.log2(2))
    return int(2**exponent)


def pad_to_fixed_length(inputs, target_length):
    """Pad the first dim in a 2D tensor, to a fixed size."""
    pad_n = int(target_length)  - inputs.shape[0]
    if pad_n < 0:
        inputs = inputs[0: int(target_length),:]
    else:
        inputs = np.pad(inputs,[[0, pad_n], [0,0]])
    return inputs


def stft_fcn(frame_size, frame_step, n_fft, window, signal):
    return librosa.stft(signal, n_fft=frame_size, hop_length=frame_step, win_length=frame_size, window=window)

def fast_spectrogram_extraction_mp(x, frame_size, frame_step, n_fft, window='hann', n_proc=8, verbose=0):
    stfts = []
    pool = mp.Pool(n_proc)
    fcn = partial(stft_fcn, frame_size, frame_step, n_fft, window)
    disable = True if verbose==0 else False
    for channel in tqdm(range(x.shape[0]), disable=disable):
        stft_channel = pool.map(fcn, x[channel,:,:])
        stfts.append(stft_channel)
    pool.close()
    return np.array(stfts)


def fast_spectrogram_extraction(x, frame_size, frame_step, n_fft, window='hann', verbose=0):
    stfts = []
#     pool = mp.Pool(n_proc)
    fcn = partial(stft_fcn, frame_size, frame_step, n_fft, window)
    disable = True if verbose==0 else False
    for channel in tqdm(range(x.shape[0]), disable=disable):
        stft_channel = []
        for instance in range(x.shape[1]):
            stft_channel.append(fcn(x[channel,instance,:]))
#         stft_channel = pool.map(fcn, x[channel,:,:])
        stfts.append(stft_channel)
#     pool.close()
    return np.array(stfts)


def retrieve_subset(x, subset_size_ratio=1):
    """Retrive a subset of x.

    Args:
        x (numpy array): the data to subsample
        subset_size (float): the proportion of the data to return. (e.g. 1 is all data)

    Returns:
        subset (numpy array): the subset
    """
    n_samples = int(subset_size_ratio * x.shape[-1])
    x = np.transpose(x, [1, 0, 2])
    x = sklearn.utils.resample(x, replace=False, n_samples=n_samples)
    x = np.transpose(x, [1, 0, 2])
    return x

# data loader to return whole spectrograms
class ClarityAudioDataloaderSequenceSpectrograms(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset="train",
                 target_length=1024,
                 new_sample_rate=8000,
                 sample_rate=44100,
                 batch_size=1,
                 frame_step=None,
                 frame_size=None,
                 spec_frame_step=None,
                 spec_frame_size=None,
                 seed=0,
                 verbose=0,
                 return_type="abs",
                 subset_size_ratio=None,
                 n_proc=8,
                 shuffling=True,
                ):
        """
        """
        random.seed(seed)

        scene_list_filename = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes.{dataset}.json"
        listener_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data"/ "metadata" / "listeners.json"
        scenes_listeners_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes_listeners.{dataset}.json"
        self.path_to_wavs = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / f"{dataset}" / "scenes"
        self.scene_list = json.load(open(scene_list_filename, "r"))
        self.listeners = json.load(open(listener_filename, "r"))
        self.scenes_listeners = json.load(open(scenes_listeners_filename, "r"))
        self.mixed_wavfiles = []
        self.target_wavfiles = []
        self.audiograms= []
        self.new_sample_rate = new_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.frame_step = frame_step
        self.spec_frame_size = spec_frame_size
        self.spec_frame_step = spec_frame_step
        self.return_type = return_type
        self.target_length = target_length
        self.eps = 1e-9
        self.verbose = verbose
        self.n_proc = n_proc
        self.subset_size_ratio = subset_size_ratio
        self.listener_ids = []
        self.scenes = []
        for scene in self.scene_list:
            target_wav_file = f"{scene['scene']}_target.wav"
            CH0 = f"{scene['scene']}_mixed_CH0.wav"
            CH1 = f"{scene['scene']}_mixed_CH1.wav"
            CH2 = f"{scene['scene']}_mixed_CH2.wav"
            CH3 = f"{scene['scene']}_mixed_CH3.wav"
            self.mixed_wavfiles.append([CH0,CH1,CH2,CH3])
            self.target_wavfiles.append(target_wav_file)
            self.scenes.append(scene)

            # for listener_name in self.scenes_listeners[scene["scene"]]:
            #     listener = self.listeners[listener_name]
            #     self.listener_ids.append(listener_name)
            #     audiogram = [listener["audiogram_levels_l"] , listener["audiogram_levels_l"]]
            #     self.audiograms.append(audiogram)
        idx = list(range(len(self.target_wavfiles)))
        self.shuffling = shuffling
        if self.shuffling:
            random.shuffle(idx)
        self.batch_idx = chunk_list(idx, batch_size)
        self.mixed_wavfiles = [self.mixed_wavfiles[i] for i in idx]
        self.target_wavfiles = [self.target_wavfiles[i] for i in idx]
        # self.audiograms = [self.audiograms[i] for i in idx]

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        y = []
        x = []
        x_specs = []
        y_specs = []
#         print(f"Grabbing {idx}")
        with tf.device('/cpu:0'):

            for isntance_n in self.batch_idx[idx]:
                # noisy signal
                # CH0 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][0]).as_posix(), self.new_sample_rate)
                CH1 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][1]).as_posix(), self.new_sample_rate)
                CH2 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][2]).as_posix(), self.new_sample_rate)
                CH3 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][3]).as_posix(), self.new_sample_rate)

                x = np.concatenate([CH1, CH2, CH3], -1)
                # x = x / np.sqrt(np.mean(x**2))
                x = pad_to_fixed_length(x, self.target_length)
                # tf.signal.stft input needs to be channel first, samples last
                x = np.transpose(x, [1,0])

                # Target
                target = load_wav_at_sample_rate((self.path_to_wavs / self.target_wavfiles[isntance_n]).as_posix(), self.new_sample_rate)
                y = pad_to_fixed_length(target, self.target_length)
                # tf.signal.stft input needs to be channel first, samples last
                y = np.transpose(y, [1,0])

                # sync the target with channel 0
                y = sync_y_to_x(x[0,:], y[0,:])
                y = np.expand_dims(y, 0)

                if self.verbose==1:
                    print("Padded audio size", x.shape, y.shape)

                # x and y are (channels, samples)
                x = frame_audio(x, self.frame_size, self.frame_step)
                y = frame_audio(y, self.frame_size, self.frame_step)

                # output is now channels, causal_steps, samples
                if self.verbose==1:
                    print("Framed+padded audio size", x.shape, y.shape)

                # Throw away a proportion of the examples
                if self.subset_size_ratio:
                    n_samples = int(self.subset_size_ratio * x.shape[1])
                    x = np.transpose(x, [1,0,2])
                    y = np.transpose(y, [1,0,2])
                    x, y = sklearn.utils.resample(x, y, replace=False, n_samples=n_samples)
                    y = np.transpose(y, [1,0,2])
                    x = np.transpose(x, [1,0,2])
                if self.n_proc > 1:
                    x_spec = fast_spectrogram_extraction_mp(x, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, n_proc=self.n_proc, verbose=self.verbose)
                    y_spec = fast_spectrogram_extraction_mp(y, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, n_proc=self.n_proc, verbose=self.verbose)
                else:
                    x_spec = fast_spectrogram_extraction(x, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, verbose=self.verbose)
                    y_spec = fast_spectrogram_extraction(y, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, verbose=self.verbose)
                del x, y, CH1, CH2, CH3
                # output is (channels, causal_steps, bins, frames)
                x_spec = np.transpose(x_spec, [1 ,3 ,2, 0])
                y_spec = np.transpose(y_spec, [1 ,3 ,2, 0])
                # output is (causal_steps, frames, bins, channels)

                if self.verbose==1:
                    print("Output shape (causal_steps, frames, bins, channels)", x_spec.shape, y_spec.shape)

                x_specs.append(x_spec)
                y_specs.append(y_spec)

            x_specs = np.concatenate(x_specs, axis=0)
            y_specs = np.concatenate(y_specs, axis=0)

            if self.return_type=="complex":
                return x_specs, y_specs
            elif self.return_type=="abs":
                return np.abs(x_specs), np.abs(y_specs)
            elif self.return_type=="abs_phase":
                return (
                    [np.abs(x_specs), np.angle(x_specs)],
                    [np.abs(y_specs), np.angle(y_specs)],
                )


class ClarityAudioDataloaderSequenceSpectrogramsEval(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset="eval",
                 target_length=1024,
                 new_sample_rate=8000,
                 sample_rate=44100,
                 batch_size=1,
                 frame_step=None,
                 frame_size=None,
                 spec_frame_step=None,
                 spec_frame_size=None,
                 seed=0,
                 verbose=0,
                 return_type="abs",
                 subset_size_ratio=None,
                 n_proc=8,
                 shuffling=True,
                 team=".E010"
                ):
        """
        """
        random.seed(seed)

        # scene_list_filename = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes.{dataset}.json"
        listener_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data"/ "metadata" / f"listeners.{dataset}.json"
        scenes_listeners_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes_listeners.{dataset}{team}.json"
        self.path_to_wavs = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / f"{dataset}" / "scenes"
        # self.scene_list = json.load(open(scene_list_filename, "r"))
        self.listeners = json.load(open(listener_filename, "r"))
        self.scenes_listeners = json.load(open(scenes_listeners_filename, "r"))
        self.mixed_wavfiles = []
        self.target_wavfiles = []
        self.audiograms= []
        self.new_sample_rate = new_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.frame_step = frame_step
        self.spec_frame_size = spec_frame_size
        self.spec_frame_step = spec_frame_step
        self.return_type = return_type
        self.target_length = target_length
        self.eps = 1e-9
        self.verbose = verbose
        self.n_proc = n_proc
        self.subset_size_ratio = subset_size_ratio
        self.scenes = []
        self.listener_ids = []
        for scene in self.scenes_listeners:
            # for listener_name in self.scenes_listeners[scene]:
            #     listener = self.listeners[listener_name]
            #     audiogram = [listener["audiogram_levels_l"] , listener["audiogram_levels_l"]]
            #     self.audiograms.append(audiogram)
            #     self.listener_ids.append(listener_name)
                # target_wav_file = f"{scene['scene']}_target.wav"
                # CH0 = f"{scene['scene']}_mixed_CH0.wav"
            CH1 = f"{scene}_mixed_CH1.wav"
            CH2 = f"{scene}_mixed_CH2.wav"
            CH3 = f"{scene}_mixed_CH3.wav"
            self.mixed_wavfiles.append([CH1,CH2,CH3])
            # self.target_wavfiles.append(target_wav_file)
            self.scenes.append(scene)
        idx = list(range(len(self.mixed_wavfiles)))
        # random.shuffle(idx)
        self.batch_idx = chunk_list(idx, batch_size)
        self.mixed_wavfiles = [self.mixed_wavfiles[i] for i in idx]
        # self.target_wavfiles = [self.target_wavfiles[i] for i in idx]
        # self.audiograms = [self.audiograms[i] for i in idx]

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        x = []
        x_specs = []
#         print(f"Grabbing {idx}")
        with tf.device('/cpu:0'):

            for isntance_n in self.batch_idx[idx]:
                # noisy signal
                CH1 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][0]).as_posix(), self.new_sample_rate)
                CH2 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][1]).as_posix(), self.new_sample_rate)
                CH3 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][2]).as_posix(), self.new_sample_rate)

                x = np.concatenate([CH1, CH2, CH3], -1)
                # x = x / np.sqrt(np.mean(x**2))
                x = pad_to_fixed_length(x, self.target_length)
                # tf.signal.stft input needs to be channel first, samples last
                x = np.transpose(x, [1,0])

                if self.verbose==1:
                    print("Padded audio size", x.shape)

                # x and y are (channels, samples)
                x = frame_audio(x, self.frame_size, self.frame_step)

                # output is now channels, causal_steps, samples
                if self.verbose==1:
                    print("Framed+padded audio size", x.shape)

                if self.n_proc > 1:
                    x_spec = fast_spectrogram_extraction_mp(x, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, n_proc=self.n_proc, verbose=self.verbose)
                else:
                    x_spec = fast_spectrogram_extraction(x, self.spec_frame_size, self.spec_frame_step, self.spec_frame_size, verbose=self.verbose)
                del x, CH1, CH2, CH3
                # output is (channels, causal_steps, bins, frames)
                x_spec = np.transpose(x_spec, [1 ,3 ,2, 0])
                # output is (causal_steps, frames, bins, channels)

                if self.verbose==1:
                    print("Output shape (causal_steps, frames, bins, channels)", x_spec.shape)

                x_specs.append(x_spec)

            x_specs = np.concatenate(x_specs, axis=0)

            if self.return_type=="complex":
                return x_specs, self.scenes[idx]
            elif self.return_type=="abs":
                return np.abs(x_specs), self.scenes[idx]
            elif self.return_type=="abs_phase":
                return [np.abs(x_specs), np.angle(x_specs)], self.scenes[idx]


class ClarityAudioDataloaderSequenceAudio(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset="train",
                 pre_padding_samples=0,
                 target_length=1024,
                 new_sample_rate=8000,
                 sample_rate=44100,
                 batch_size=1,
                 frame_step=None,
                 frame_size=None,
                 seed=0,
                 n_mels=128,
                 verbose=0,
                 return_type="abs",
                 shuffling=True,
                ):
        """
        """
        random.seed(seed)

        scene_list_filename = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes.{dataset}.json"
        listener_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data"/ "metadata" / "listeners.json"
        scenes_listeners_filename =  pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / "metadata" / f"scenes_listeners.{dataset}.json"
        self.path_to_wavs = pathlib.Path(PATH_TO_CLARITY_FOLDER) / "data" / "clarity_data" / f"{dataset}" / "scenes"
        self.scene_list = json.load(open(scene_list_filename, "r"))
        self.listeners = json.load(open(listener_filename, "r"))
        self.scenes_listeners = json.load(open(scenes_listeners_filename, "r"))
        self.mixed_wavfiles = []
        self.target_wavfiles = []
        self.audiograms= []
        self.pre_padding_samples = pre_padding_samples
        self.new_sample_rate = new_sample_rate
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.frame_step = frame_step
        self.return_type = return_type

        self.target_length = target_length
        self.eps = 1e-9
        self.verbose = verbose
        for scene in self.scene_list:
            for listener_name in self.scenes_listeners[scene["scene"]]:
                listener = self.listeners[listener_name]
                audiogram = [listener["audiogram_levels_l"] , listener["audiogram_levels_l"]]
                target_wav_file = f"{scene['scene']}_target.wav"
                CH0 = f"{scene['scene']}_mixed_CH0.wav"
                CH1 = f"{scene['scene']}_mixed_CH1.wav"
                CH2 = f"{scene['scene']}_mixed_CH2.wav"
                CH3 = f"{scene['scene']}_mixed_CH3.wav"
                self.mixed_wavfiles.append([CH0,CH1,CH2,CH3])
                self.target_wavfiles.append(target_wav_file)
                self.audiograms.append(audiogram)
        idx = list(range(len(self.target_wavfiles)))
        if shuffling:
            random.shuffle(idx)
        self.batch_idx = chunk_list(idx, batch_size)
        self.mixed_wavfiles = [self.mixed_wavfiles[i] for i in idx]
        self.target_wavfiles = [self.target_wavfiles[i] for i in idx]
        self.audiograms = [self.audiograms[i] for i in idx]

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        xs = []
        ys = []
        for isntance_n in self.batch_idx[idx]:
            # noisy signal
            CH1 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][1]).as_posix(), self.new_sample_rate)
            CH2 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][2]).as_posix(), self.new_sample_rate)
            CH3 = load_wav_at_sample_rate((self.path_to_wavs / self.mixed_wavfiles[isntance_n][3]).as_posix(), self.new_sample_rate)
            x = tf.concat([CH1, CH2, CH3], -1)

            # Target
            y = load_wav_at_sample_rate(self.path_to_wavs / self.target_wavfiles[isntance_n], self.new_sample_rate)
            if self.verbose==1:
                print("Padded audio size", x.shape, y.shape)
            xs.append(x)
            ys.append(y)

        xs = tf.stack(xs)
        ys = tf.stack(ys)

        return xs, ys
