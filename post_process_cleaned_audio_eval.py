import sys
import numpy as np
import soundfile as sf
import scipy
from scipy import signal
from plotly import graph_objects as go
import librosa
import tqdm
from argparse import ArgumentParser
import os
from pathlib import Path
import json

def eq_signals(
    x_L,
    x_R,
    audiogram_levels_l,
    audiogram_levels_r,
    audiogram_cfs,
    max_gain=30,
    fs=44100,
    window_len=100,
    zero_phase=True,
    verbose=0,
    max_valid_freq=7500,
    min_valid_freq=1600,

):
    """Hearing aid eq.

    Args:
        x_L (numpy array): left ear signal
        x_R (numpy array): right ear signal
        audiogram_levels_l (list): pure tone response level, left ear
        audiogram_levels_r (list): pure tone response level, right ear
        audiogram_cfs (list): center frequencies for the pure tone audiogram
        max_gain (float): maximum gain to apply
        fs(int): sample freq
        window_len (int):
        zero_phase (bool): if true the first window_len //2 samples at the
            output will be thrown away, making the result zero phase.
            This means the resulting filter looks ahead by window_len //2 samples

    Returns:
        out_L (numpy array): equalised left ear
        out_R (numpy array): equalised right ear
        max_gain_L (float): max gain applied to left ear
        max_gain_R (float): max gain applied to right ear
    """
    gains_L = []
    gains_R = []
    filters = []
    # pad the end of each signal, so that signal can be truncated to same length as input on return
    pad_ammount = window_len//2
    if zero_phase:
        x_L = np.pad(x_L, [0, pad_ammount])
        x_R = np.pad(x_R, [0, pad_ammount])

    # with an FIR filter, low frequencies cannot be effected when filter taps are not long enough
    # filters with a centre freq below this limit are left out of the filter bank
    # min_valid_freq = 1/(window_len / fs)
    i_valid_freq_indexes = [n for n, fc in enumerate(audiogram_cfs) if fc >= min_valid_freq]
    audiogram_levels_l_ = [audiogram_levels_l[n] for n in i_valid_freq_indexes]
    audiogram_levels_r_ = [audiogram_levels_r[n] for n in i_valid_freq_indexes]
    audiogram_cfs_ = [audiogram_cfs[n] for n in i_valid_freq_indexes]

    # Any gains for filters below this freq limit are collected together into a single value and the powers averaged
    # The averaging incluedes the lower valid frequecy band, this gain value then becomes the gain of the
    # low pass filter in the filter bank.
    # inother words the bottom bands are all condensed into one
    average_power_lower_bands_L = np.mean(np.power(10, np.array(audiogram_levels_l)[0:i_valid_freq_indexes[0]+1] / 10))
    average_power_lower_bands_R = np.mean(np.power(10, np.array(audiogram_levels_r)[0:i_valid_freq_indexes[0]+1] / 10))
    audiogram_levels_l_[0] = 10*np.log10(average_power_lower_bands_L)
    audiogram_levels_r_[0] = 10*np.log10(average_power_lower_bands_R)

    # if the highest centre freq is invalid, lump this into the penultimate and make this the higest
    i_valid_freq_indexes = [n for n, fc in enumerate(audiogram_cfs_) if fc <= max_valid_freq]
    average_power_higher_bands_L = np.mean(np.power(10, np.array(audiogram_levels_l_)[i_valid_freq_indexes[-1]:] / 10))
    average_power_higher_bands_R = np.mean(np.power(10, np.array(audiogram_levels_r_)[i_valid_freq_indexes[-1]:] / 10))
    audiogram_levels_l_ = [audiogram_levels_l_[n] for n in i_valid_freq_indexes]
    audiogram_levels_r_ = [audiogram_levels_r_[n] for n in i_valid_freq_indexes]
    audiogram_cfs_ = [audiogram_cfs_[n] for n in i_valid_freq_indexes]
    audiogram_levels_l_[-1] = 10*np.log10(average_power_higher_bands_L)
    audiogram_levels_r_[-1] = 10*np.log10(average_power_higher_bands_R)
    # using the audio gram data calculate gain values to apply for each filter
    # find the pure tone level for the best freq band in each ear
    # best_L = min(audiogram_levels_l)
    # best_R = min(audiogram_levels_r)

    #
    best_L = min(audiogram_levels_r_+audiogram_levels_l_)
    best_R = min(audiogram_levels_r_+audiogram_levels_l_)
    for loss_L, loss_R, fc in zip(audiogram_levels_l_, audiogram_levels_r_, audiogram_cfs_):
        # the gain to apply. either the difference between the pure tone level and the best pure tone level
        # or if this is greater than the max gain, the max gain.
        gain_L = np.min((max_gain, loss_L-best_L))
        gain_R = np.min((max_gain, loss_R-best_R))
        gains_L.append(gain_L)
        gains_R.append(gain_R)

    # build filter bank
    nyq = 0.5 * fs
    # 1st filter is a low pass filter, cut off half way to next filter
    high = (audiogram_cfs_[0] + (audiogram_cfs_[1] - audiogram_cfs_[0])/2) / nyq
    # filter = signal.butter(order, [40/nyq, high], btype='band', output='sos')
    filter = signal.firwin(window_len, [40/nyq, high], pass_zero=False)
    # sos = signal.butter(order, low, btype='low', output='sos')
    filters.append(filter)
    # define all band pass filters
    for n in range(1, len(audiogram_cfs_)-1):
        fc = audiogram_cfs_[n]
        lowcut = fc - (audiogram_cfs_[n] - audiogram_cfs_[n-1])/2
        highcut = fc + (audiogram_cfs_[n+1] - audiogram_cfs_[n])/2
        low = lowcut / nyq
        high = highcut / nyq
        # filter = signal.butter(order, [low, high], btype='band', output='sos')
        filter = signal.firwin(window_len, [low, high], pass_zero=False)
        filters.append(filter)

    # last filter is high pass
    low = (audiogram_cfs_[-2] + (audiogram_cfs_[-1] - audiogram_cfs_[-2])/2) / nyq
    # filter = signal.butter(order*2, high, btype='high', output='sos')
    filter = signal.firwin(window_len, [low, 0.999], pass_zero=False)
    filters.append(filter)
    # filter the signal
    out_L = []
    out_R = []
    if verbose:
        fig1 = go.Figure()
        fig2 = go.Figure()
    for gain_L, gain_R, filter in zip(gains_L, gains_R, filters):
        # filter then apply approprate gain
        y_L = signal.lfilter(filter, 1.0, x_L) * np.power(10, gain_L/20)
        y_R = signal.lfilter(filter, 1.0, x_R) * np.power(10, gain_R/20)
        # y_L = signal.sosfilt(filter, x_L) * np.power(10, gain_L/20)
        # y_R = signal.sosfilt(filter, x_R) * np.power(10, gain_R/20)

        if verbose:
            # y_sos = signal.sosfilt(filter, x)
            # w, h = signal.sosfreqz(filter,worN=10000)
            w, h = signal.freqz(filter)
            fig1.add_trace(go.Scatter(x=w, y=20*np.log10(abs(h* np.power(10, gain_L/20)))))
            fig2.add_trace(go.Scatter(x=w, y=20*np.log10(abs(h* np.power(10, gain_R/20)))))

        out_L.append(y_L)
        out_R.append(y_R)
    if verbose:
        fig1.show()
        fig2.show()
    # reconstruct broadband signal
    out_L = np.sum(out_L, 0)
    out_R = np.sum(out_R, 0)
    if zero_phase:
        out_L = out_L[pad_ammount:]
        out_R = out_R[pad_ammount:]
    return out_L, out_R, max(gains_L), max(gains_R)


def soft_clip(x, clip_limit=1):
    """Implementation of a cubic soft-clipper
    https://ccrma.stanford.edu/~jos/pasp/Cubic_Soft_Clipper.html
    """
    deg = 21

    maxamp = np.max(abs(x))

    if maxamp < clip_limit:
        return x
    elif maxamp >= clip_limit:
        xclipped = np.where(
            x > clip_limit,
            (deg - 1) / deg,
            np.where(x < -clip_limit, -(deg - 1) / deg, x - x ** deg / deg),
        )
        return xclipped


def compressor_twochannel(x, Fs, T, R, attackTime, releaseTime):
    """
    This function implements a two channel compressor where the
    same scaling is applied to both channels. This function is based on
    the standard MATLAB dynamic range compressor [1] and its "Hack
    Audio" implementation: https://www.hackaudio.com/

    [1] Giannoulis, Dimitrios, Michael Massberg, and Joshua D. Reiss. "Digital
    Dynamic Range Compressor Design: A Tutorial and Analysis." Journal of
    Audio Engineering Society. Vol. 60, Issue 6, 2012, pp. 399-408.


    Args:
        x (ndarray): signal.
        Fs (int): sampling rate.
        T (int): threshold relative to 0 dBFS.
        R (int): compression ratio.
        attackTime (float): attack time in seconds.
        releaseTime (float): release time in seconds.

    Returns:
        ndarray: signal y


    """
    N = len(x)
    channels = np.shape(x)[1]
    if channels != 2:
        raise ValueError("Channel mismatch.")
    y = np.zeros((N, 2))
    lin_A = np.zeros((N, 1))

    # Get attack and release times
    alphaA = np.exp(-np.log(9) / (Fs * attackTime))
    alphaR = np.exp(-np.log(9) / (Fs * releaseTime))

    gainSmoothPrev = 0  # Initialise smoothed gain variable

    # Loop over each sample
    for n in range(N):
        # Derive dB of sample x[n]
        xn_left = np.abs(x[n, 0])
        xn_right = np.abs(x[n, 1])
        xn = max(xn_left, xn_right)
        with np.errstate(divide="ignore"):
            x_dB = 20 * np.log10(np.divide(xn, 1))

        # Ensure there are no values of negative infinity
        if x_dB < -96:
            x_dB = -96

        # Check if sample is above threshold T
        # Static Characteristic - applying hard knee
        if x_dB > T:
            gainSC = T + (x_dB - T) / R  # Perform compression
        else:
            gainSC = x_dB  # No compression

        # Compute the gain change as the difference
        gainChange_dB = gainSC - x_dB

        # Smooth the gain change using the attack and release times
        if gainChange_dB < gainSmoothPrev:
            # Attack
            gainSmooth = ((1 - alphaA) * gainChange_dB) + (alphaA * gainSmoothPrev)
        else:
            # Release
            gainSmooth = ((1 - alphaR) * gainChange_dB) + (alphaR * gainSmoothPrev)

        # Translate the gain to the linear domain
        lin_A[n, 0] = 10 ** (np.divide(gainSmooth, 20))

        # Apply linear amplitude to input sample
        y[n, 0] = lin_A[n, 0] * x[n, 0]
        y[n, 1] = lin_A[n, 0] * x[n, 1]

        # Update smoothed gain
        gainSmoothPrev = gainSmooth

    return y


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "-d",
        type=str,
        dest="dataset",
        help="Dataset train eval or dev.",
        default="eval2",
    )
    ap.add_argument(
        "-i",
        type=str,
        dest="input_path",
        help="Location of the cleaned signals.",
        default="~/data/submission/cleaned_scenes/eval",

    )
    ap.add_argument(
        "-o",
        type=str,
        dest="output_path",
        help="Where to write the outputs to.",
        # default="~/data/submission/cleaned_scenes/eval",
        default="/home/kenders/greenhdd/clarity_challenge/data/listening_test_data"
    )
    ap.add_argument(
        "-s",
        type=str,
        dest="clarity_data",
        help="Location of clarity dataset.",
        # default="~/data/Data/clarity_CEC1_data/clarity_data/",
        default="/home/kenders/greenhdd/clarity_challenge/data/clarity_CEC1_data/clarity_data/"
    )



    args = ap.parse_args()
    dataset = args.dataset
    input_path = Path(args.input_path).expanduser().as_posix()
    output_path = Path(args.output_path).expanduser().as_posix()
    clarity_data = Path(args.clarity_data).expanduser().as_posix()
    if "CLARITY_ROOT" in os.environ:
        CLARITY_ROOT = os.environ['CLARITY_ROOT']
        # CLARITY_DATA = (Path(CLARITY_ROOT) / "data/clarity_data")
        # scene_list_filename = f"{clarity_data}/metadata/scenes.{dataset}.json"
        listener_filename = f"{clarity_data}/metadata/listeners.{dataset}.json"
        scenes_listeners_filename = f"{clarity_data}/metadata/scenes_listeners.{dataset}.E010.json"
        # input_path = f"{CLARITY_DATA}/{dataset}/scenes"
        # output_path = f"{CLARITY_DATA}/{dataset}/scenes"
    else:
        raise RuntimeError("CLARITY_ROOT environment var not set")

    # from `run_HA_processing.py`
    # scene_list = json.load(open(scene_list_filename, "r"))
    listeners = json.load(open(listener_filename, "r"))
    scenes_listeners = json.load(open(scenes_listeners_filename, "r"))
    for scene_n, scene in tqdm.tqdm(enumerate(scenes_listeners), total=len(scenes_listeners)):
    # for scene_n, scene in tqdm.tqdm(enumerate(scene_list), total=len(scene_list)):
    #     for listener_name in scenes_listeners[scene["scene"]]:
        listener_names = scenes_listeners[scene]
        for listener_name in listener_names:
            listener = listeners[listener_name]
            cleaned_file = Path(input_path) / f"{scene}_cleaned_signal_16k.wav"
            if not cleaned_file.exists():
                raise RuntimeError()
            else:
                ######################################################################
                # custom hearing aid (process at 16 k)
                window_len = 11
                audio, fs_hearing_aid = sf.read(cleaned_file)
                fil_audio_L, fil_audio_R, max_gain_L, max_gain_R = eq_signals(
                    audio[:,0], audio[:,1],
                    listener["audiogram_levels_l"], listener["audiogram_levels_r"], listener["audiogram_cfs"],
                    max_gain = 30, fs = fs_hearing_aid, window_len = window_len,
                    max_valid_freq=7000,
                    min_valid_freq=1500,
                )
                fil_audio = np.stack([fil_audio_L, fil_audio_R], 1)
                fil_audio_compressed = compressor_twochannel(fil_audio, fs_hearing_aid, T=-6, R=5, attackTime=4e-3, releaseTime=75-3)
                fil_audio_compressed[:,0] = soft_clip(fil_audio_compressed[:,0], clip_limit=0.99999)
                fil_audio_compressed[:,1] = soft_clip(fil_audio_compressed[:,1], clip_limit=0.99999)
                # print(10 * np.log10(np.mean(fil_audio_compressed**2,0))+120)

                # load the origianl channel 1 to ensure the output lenght is the same (truncate the zero padding)
                original_ch1, fs = sf.read(Path(clarity_data) / dataset / "scenes" / f"{scene}_mixed_CH1.wav")
                original_samples = original_ch1.shape[0]
                fil_audio_compressed_44k = librosa.resample(fil_audio_compressed.T, fs_hearing_aid, 44100)
                fil_audio_compressed_44k = fil_audio_compressed_44k[:,0:original_samples].T
                sf.write(f"{output_path}/{scene}_{listener_name}_HA-output.wav", fil_audio_compressed_44k, 44100, subtype="FLOAT")
                # check the output is syncronised with the input (CH1, left ear)
                corr = signal.correlate(original_ch1[:, 0], fil_audio_compressed_44k[: ,0], mode='full', method='auto')
                in2_len = fil_audio_compressed_44k.shape[0]
                in1_len = original_ch1.shape[0]
                lags = np.arange(-in2_len + 1, in1_len)
                delay = lags[np.argmax(np.abs(corr))]
                if delay != 0:
                    print(f"Left ear Signal is not properly synced with input {scene} {listener_name} delay {delay}")
                # check the output is syncronised with the input (CH1, right ear)
                corr = signal.correlate(original_ch1[:, 1], fil_audio_compressed_44k[: ,1], mode='full', method='auto')
                in2_len = fil_audio_compressed_44k.shape[0]
                in1_len = original_ch1.shape[0]
                lags = np.arange(-in2_len + 1, in1_len)
                delay = lags[np.argmax(np.abs(corr))]
                if delay != 0:
                    print(f"Right ear Signal is not properly synced with input {scene} {listener_name} delay {delay}")
