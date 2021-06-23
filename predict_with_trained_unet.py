from argparse import ArgumentParser
from dataloading import ClarityAudioDataloaderSequenceSpectrogramsEval, ClarityAudioDataloaderSequenceSpectrograms, ClarityAudioDataloaderSequenceAudio
import sklearn
import tensorflow as tf
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import tqdm
from custom_unet import custom_unet
from pathlib import Path
import os
import soundfile as sf
import IPython.display as ipd
import plotly.express as px
from scipy import signal
import pandas as pd
import tqdm
from functools import partial
import multiprocessing as mp
import pathlib



def chunk_list(input_list, chunk_size):
    """Convert list into a list of lists."""
    return [
        input_list[chunk : chunk + chunk_size]
        for chunk in range(0, len(input_list), chunk_size)
    ]

def overlap_reconstruct(
    out,
    lookahead_frame_size,
    lookahead_frame_step,
    reconstruction_overlap,
):
    """Overlap and add frames of audio to reconstruct.

    Args:
        out (list): list of frames of audio (e.g. 80 samples long each)
        lookahead_frame_size (int): the size of the window used in the CNN e.g. 6 s
        lookahead_frame_step (int): the look ahead ammount (e.g. 80 samples)
        reconstruction_overlap (int): Ammount each frame is overlap added (e.g. 10 samples)

    Returns:
        reconstructed_audio (np.array): lookahead_frame_size long
    """
    reconstructed_audio = np.zeros(len(out)*lookahead_frame_step)
    current_frame_smoothed = []
    previous_frame_smoothed = out[0]
    for n in range(1, len(out)):
        current_frame = out[n]
        # window to mix the starting reconstruction_overlap samples of one frame with the
        # end reconstruction_overlap samples of the next
        hanning_window = np.hanning(reconstruction_overlap*2)
        current_frame[0:reconstruction_overlap] *= hanning_window[0:reconstruction_overlap]
        current_frame[len(current_frame)-reconstruction_overlap:] *= hanning_window[reconstruction_overlap:]
        start_of_frame = (n-1) * (lookahead_frame_step)
        reconstructed_audio[start_of_frame:(start_of_frame+lookahead_frame_step+reconstruction_overlap)] += current_frame
    # remove the extra `reconstruction_overlap` samples that were grabbed from the first frame
    return reconstructed_audio[reconstruction_overlap:]


def reconstruct_cleaned_audio(
        x_spec,
        spec_frame_size,
        spec_frame_step,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
        verbose=0,
         n_proc=8,


):
    """Reconstruct cleaned audio from clean spectrograms.
    Note this requires a global keras model `model` (faster than passing as an argument.)

    Args:
        x_spec (numpy or tf array): the processed spectrograms [causal_frames, spec_frames, bins, channels]
        spec_frame_size (int): the spectrogram frame size in samples
        spec_frame_step (int): the spectrogram frame step in samples
        lookahead_frame_size (int): the size of the audio window captured for each spectrogram (e.g. 6 * fs)
        lookahead_frame_step (int): the step in samples that the audio window slid over the input signal (e.g. 40 samples)
        verbose (int): display a progress bar if 1

    Returns:
        audio (np.array): the reconstructed audio (left ear, relative to channel 0 which is channel 1 in the clarity doc))
    """
    out = []
    display_progress_bar = False if verbose == 1 else True

    for x in tqdm.tqdm(x_spec, disable=display_progress_bar):
        x_cleaned_abs = model.predict(tf.expand_dims(tf.math.abs(x), 0))[0,...,0]
        # grab the noisy phase from Ch1 left ear
        x_phase_noisy = np.angle(x[...,0])
        x_cleaned_complex = x_cleaned_abs * (np.cos(x_phase_noisy) + np.sin(x_phase_noisy)*1.j)

        # istft using librosa
        x_cleaned_complex = np.transpose(x_cleaned_complex, [1, 0])
        # Luckily because the  audio length (96000) is an integer multiple of the hop size(256)
        # no padding of the signal is required, this is just pure luck!!
        reconstructed = librosa.istft(x_cleaned_complex, hop_length=spec_frame_step, win_length=spec_frame_size, length=lookahead_frame_size)
        # grab the last `lookahead_frame_step
        out.append(reconstructed[(lookahead_frame_size-lookahead_frame_step-reconstruction_overlap):lookahead_frame_size])
    out = overlap_reconstruct(
        out,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
    )
    return out

def infinite_sequence():
    num = 0
    while True:
        yield num


def reconstruct_cleaned_audio_mp(
        x_spec,
        spec_frame_size,
        spec_frame_step,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
        verbose=0,
        n_proc=8
):
    """Reconstruct cleaned audio from clean spectrograms.

    Note this requires a global keras model `model` (faster than passing as an argument.)

    Args:
        x_spec (numpy or tf array): the processed spectrograms [causal_frames, spec_frames, bins, channels]
        spec_frame_size (int): the spectrogram frame size in samples
        spec_frame_step (int): the spectrogram frame step in samples
        lookahead_frame_size (int): the size of the audio window captured for each spectrogram (e.g. 6 * fs)
        lookahead_frame_step (int): the step in samples that the audio window slid over the input signal (e.g. 40 samples)
        verbose (int): display a progress bar if 1

    Returns:
        audio (np.array): the reconstructed audio (left ear, relative to channel 0 which is channel 1 in the clarity doc))
    """
    out = []
    display_progress_bar = False if verbose == 1 else True
    x_spec_chunked = chunk_list(x_spec, n_proc)
    with mp.Pool(n_proc) as pool:
        fcn = partial(_reconstruct, spec_frame_size, spec_frame_step, lookahead_frame_size, lookahead_frame_step, reconstruction_overlap)
        for x_spec_chunk in tqdm.tqdm(x_spec_chunked, disable=display_progress_bar):
            x_cleaned_abs = model.predict(x_spec_chunk)[...,0]
            x_phase_noisy = np.angle(x_spec_chunk[...,0])
            x_cleaned_complex = x_cleaned_abs * (np.cos(x_phase_noisy) + np.sin(x_phase_noisy)*1.j)
            out.extend(pool.map(fcn, x_cleaned_complex))

    out = overlap_reconstruct(
        out,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
    )

    return out

def _reconstruct(
    spec_frame_size,
    spec_frame_step,
    lookahead_frame_size,
    lookahead_frame_step,
    reconstruction_overlap,
    x_cleaned_complex,
):
    # istft using librosa
    x_cleaned_complex = np.transpose(x_cleaned_complex, [1, 0])
    reconstructed = librosa.istft(
        x_cleaned_complex,
        hop_length=spec_frame_step,
        win_length=spec_frame_size,
        length=lookahead_frame_size
    )
    return reconstructed[(lookahead_frame_size-lookahead_frame_step-reconstruction_overlap):lookahead_frame_size]



def reconstruct_original_audio(
        x_spec,
        spec_frame_size,
        spec_frame_step,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
        verbose=0
):
    """Reconstruct original audio from clean spectrograms.

    Args:
        x_spec (numpy or tf array): the processed spectrograms [causal_frames, spec_frames, bins, channels]
        spec_frame_size (int): the spectrogram frame size in samples
        spec_frame_step (int): the spectrogram frame step in samples
        lookahead_frame_size (int): the size of the audio window captured for each spectrogram (e.g. 6 * fs)
        lookahead_frame_step (int): the step in samples that the audio window slid over the input signal (e.g. 40 samples)
        verbose (int): display a progress bar if 1

    Returns:
        audio (np.array): the reconstructed audio (left ear, relative to channel 0 which is channel 1 in the clarity doc))
    """
    out = []
    for x in tqdm.tqdm(x_spec):
        x_abs = np.abs(x)[...,0]
        x_phase_noisy = np.angle(x[...,0])
        x_complex = x_abs * (np.cos(x_phase_noisy) + np.sin(x_phase_noisy)*1.j)

        # if using librosa
        x_complex = np.transpose(x_complex, [1, 0])
        reconstructed = librosa.istft(x_complex, hop_length=spec_frame_step, win_length=spec_frame_size, length=lookahead_frame_size)
        out.append(reconstructed[(lookahead_frame_size-lookahead_frame_step):lookahead_frame_size])
    out = overlap_reconstruct(
        out,
        lookahead_frame_size,
        lookahead_frame_step,
        reconstruction_overlap,
    )
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "-p",
        type=str,
        dest="path_to_model",
        help="Path a a trained keras model.",
        default="/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/models/9_0.01611/",
        # default="/home/ubuntu/spectrogram_based_model/models/9_0.01611/",
        # default="/home/paulkendrick/spectrogram_based_models/models/9_0.01611/",
    )
    ap.add_argument(
        "-d",
        type=str,
        dest="dataset",
        help="Dataset train eval or dev.",
        default="eval2",
    )
    args = ap.parse_args()
    dataset = args.dataset
    eps = 1e-9
    fs = 16000
    output_path = pathlib.Path(os.environ['CLARITY_ROOT']) / "data" / "clarity_data"
    spec_frame_size = 1024 # odd fft ensures even rfft
    spec_frame_step = spec_frame_size // 4 #int(fs * 5e-3) // 2

    channels_in = 6
    # the input processing window size
    lookahead_frame_size = int(6.0*fs)
    # the maximum lookahead in the clarity challenge rules
    causal_buffer = int(np.floor(5e-3 * fs))
    # samples to overlap each buffer
    reconstruction_overlap = 10
    # Hop size for input frames
    lookahead_frame_step = int(causal_buffer) - reconstruction_overlap
    batch_size = 1
    print("Loading model")
    global model
    model = tf.keras.models.load_model(args.path_to_model)
    # model = tf.keras.models.load_model("/home/ubuntu/spectrogram_based_model/models/9_0.01611/")

    model.summary()
    multi_proc = False
    verbose = 0
    n_proc = 1

    # for the train and dev sets usinf ClarityAudioDataloaderSequenceSpectrograms
    if dataset == "dev" or dataset =="train":
        data_loader = ClarityAudioDataloaderSequenceSpectrograms(
            dataset=dataset,
            spec_frame_step=spec_frame_step,
            spec_frame_size=spec_frame_size,
            frame_step=lookahead_frame_step,
            frame_size=lookahead_frame_size,
            new_sample_rate=fs,
            batch_size=batch_size,
            target_length=6.0*fs,
            verbose=0,
            return_type="complex",
            n_proc=1,
            shuffling=False,
            audio_path_dataset_name=dataset
        )
        # test the dataloader
        # x_spec, y_spec = data_loader[0]
        # frames = x_spec.shape[1]
        # bins = x_spec.shape[2]
        # channels = x_spec.shape[3]
    elif dataset == "eval" or dataset == "eval2":
        # just when using the eval set (as there are no targets) usding
        # ClarityAudioDataloaderSequenceSpectrogramsRval
        # dataset = "eval"
        data_loader = ClarityAudioDataloaderSequenceSpectrogramsEval(
            dataset=dataset,
            spec_frame_step=spec_frame_step,
            spec_frame_size=spec_frame_size,
            frame_step=lookahead_frame_step,
            frame_size=lookahead_frame_size,
            new_sample_rate=fs,
            batch_size=batch_size,
            target_length=6.0*fs,
            verbose=0,
            return_type="complex",
            n_proc=1,
            shuffling=False,
            audio_path_dataset_name="eval"
        )
    else:
        raise RuntimeError("Unsupported dataset")

    # using the OrderedEnqueuer to iterate the keras sequence enables pre fetching
    # however this requires a huge ammount of memory as two instances are required in memory
    # (each is around 22gb)
    # enq = tf.keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=False)
    # enq.start(workers=1)
    # gen = enq.get()
    # progbar.add(len(data_loader)//2)
    n = 0
    progbar = tf.keras.utils.Progbar(len(data_loader) -n)
    for batch_n in range(n, len(data_loader)):
        # when using prefetching (uses too much memory)
        # x_spec, scene = next(gen)

        # for the eval set only, using ClarityAudioDataloaderSequenceSpectrogramsEval
        # x_spec, scene = data_loader[batch_n]

        # for the dev and train set, using ClarityAudioDataloaderSequenceSpectrograms
        # assert data_loader.shuffling == False, "Shuffling must be disabled"
        # if multi_proc:
        #     x_spec, scene = next(gen)
        # else:
        x_spec, y_spec = data_loader[batch_n]

        scene = data_loader.scenes[batch_n]
        if multi_proc:
            reconstructed_audio_full_L = reconstruct_cleaned_audio_mp(x_spec, spec_frame_size, spec_frame_step, lookahead_frame_size, lookahead_frame_step, reconstruction_overlap, verbose=verbose, n_proc=n_proc)
        else:
            reconstructed_audio_full_L = reconstruct_cleaned_audio(x_spec, spec_frame_size, spec_frame_step, lookahead_frame_size, lookahead_frame_step, reconstruction_overlap, verbose=verbose, n_proc=n_proc)
        # mirror the head, i.e. swap the ears over and now the right ear channel 1 is rhe reference.
        x_spec_flip = x_spec[..., [1,0,3,2,5,4]]
        if multi_proc:
            reconstructed_audio_full_R = reconstruct_cleaned_audio_mp(x_spec_flip, spec_frame_size, spec_frame_step, lookahead_frame_size, lookahead_frame_step, reconstruction_overlap, verbose=verbose, n_proc=n_proc)
        else:
            reconstructed_audio_full_R = reconstruct_cleaned_audio(x_spec_flip, spec_frame_size, spec_frame_step, lookahead_frame_size, lookahead_frame_step, reconstruction_overlap, verbose=verbose, n_proc=n_proc)
        reconstructed_audio_full = np.stack([reconstructed_audio_full_L, reconstructed_audio_full_R], axis=1)

        output_filename = f"{output_path}/eval/{scene}_cleaned_signal_16k.wav"
        # output_filename = f"{output_path}/{dataset}/{scene['scene']}_cleaned_signal_16k.wav"
        sf.write(output_filename, reconstructed_audio_full, fs)

        # reconstructed_audio_full_44k = librosa.resample(reconstructed_audio_full.T, fs, 44100)
        # output_filename_44k = f"{output_path}/{dataset}/{scene['scene']}_cleaned_signal_441k.wav"
        # sf.write(output_filename_44k, reconstructed_audio_full_44k.T, 44100)
        progbar.add(1)


if __name__ ==  "__main__":
    main()
