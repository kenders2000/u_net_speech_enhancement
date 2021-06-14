# $PYTHON_BIN "$CLARITY_ROOT"/scripts/run_HA_processing.py  --num_channels "$num_channels" "$CLARITY_DATA"/metadata/scenes."$dataset".json "$CLARITY_DATA"/metadata/listeners.json "$CLARITY_DATA"/metadata/scenes_listeners."$dataset".json "$CLARITY_DATA"/"$dataset"/scenes "$CLARITY_DATA"/"$dataset"/scenes
## Set these paths
CLARITY_ROOT = "/home/paulkendrick/clarity_CEC1"
CLARITY_DATA = "/home/paulkendrick/clarity_CEC1/data/clarity_data"
CFG_TEMPLATE = "/home/paulkendrick/spectrogram_based_models/modified_prerelease_combination4_smooth_template.cfg"


########
import sys
sys.path.append('/home/paulkendrick/clarity_CEC1/env/lib/python3.6/site-packages')
# ls /home/paulkendrick/clarity_CEC1/env/lib/python3.6/site-packages/audio_dspy
import audio_dspy as adsp
import numpy as np
import soundfile as sf
import scipy
from scipy import signal
from plotly import graph_objects as go
import librosa

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
    print("lowpass", high*nyq)
    # define all band pass filters
    for n in range(1, len(audiogram_cfs_)-1):
        fc = audiogram_cfs_[n]
        lowcut = fc - (audiogram_cfs_[n] - audiogram_cfs_[n-1])/2
        highcut = fc + (audiogram_cfs_[n+1] - audiogram_cfs_[n])/2
        low = lowcut / nyq
        high = highcut / nyq
        # filter = signal.butter(order, [low, high], btype='band', output='sos')
        filter = signal.firwin(window_len, [low, high], pass_zero=False)
        print("bandpass",low*nyq,fc, high*nyq)
        filters.append(filter)

    # last filter is high pass
    low = (audiogram_cfs_[-2] + (audiogram_cfs_[-1] - audiogram_cfs_[-2])/2) / nyq
    # filter = signal.butter(order*2, high, btype='high', output='sos')
    filter = signal.firwin(window_len, [low, 0.999], pass_zero=False)
    filters.append(filter)
    print("highpass", low*nyq)
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

#############################################################################
# doing what run_HA_processing.py does
# copy the 2 channel template into the GHA
CFG_TEMPLATE_TO = f"{CLARITY_ROOT}/projects/GHA/cfg_files"
import shutil
import os
from pathlib import Path

shutil.copy(CFG_TEMPLATE, CFG_TEMPLATE_TO)
dataset = "dev"
scene_list_filename = f"{CLARITY_DATA}/metadata/scenes.{dataset}.json"
listener_filename = f"{CLARITY_DATA}/metadata/listeners.json"
scenes_listeners_filename = f"{CLARITY_DATA}/metadata/scenes_listeners.{dataset}.json"
input_path = f"{CLARITY_DATA}/{dataset}/scenes"
output_path = f"{CLARITY_DATA}/{dataset}/scenes"

# output_path = "/tmp/clarity_test"
if not Path(output_path).exists:
    os.makedirs(output_path)
channels = 1

import argparse
from pathlib import Path
import logging
import json
from tqdm import tqdm
import sys
from clarity_core.config import CONFIG
import tempfile
import numpy as np
import subprocess

sys.path.append(r"../projects/GHA")
from GHA import GHAHearingAid as HearingAid
import GHA
import clarity_core.signal as ccs


def create_HA_inputs(infile_names, merged_filename):
    """Create input signal for the baseline hearing aids.

    This replaces the clarity_core.signal.create_HA_inputs function. So that
    the inputs can be easily set.

    """

    # if (infile_names[0][-5] != "1") or (infile_names[2][-5] != "3"):
    #     raise Exception("HA-input signal error: channel mismatch!")

    signal_CH1 = ccs.read_signal(infile_names[0])
    # signal_CH3 = ccs.read_signal(infile_names[2])

    merged_signal = np.zeros((len(signal_CH1), 4))
    merged_signal[:, 0] = signal_CH1[
        :, 0
    ]  # channel index 0 = front microphone on the left hearing aid
    merged_signal[:, 1] = signal_CH1[
        :, 1
    ]  # channel index 1 = front microphone on the right hearing aid
    merged_signal[:, 2] = signal_CH1[
        :, 0
    ]  # channel index 0 = rear microphone on the left hearing aid
    merged_signal[:, 3] = signal_CH1[
        :, 1
    ]  # channel index 1 = rear microphone on the right hearing aid


    ccs.write_signal(merged_filename, merged_signal, CONFIG.fs, floating_point=True)


def process_files(self, infile_names, outfile_name):
    """Process a set of input signals and generate an output.
    Replaces GHA.GHAHearingAid.process_files using monkey patching
    Args:
        infile_names (list[str]): List of input wav files. One stereo wav
            file for each hearing device channel
        outfile_name (str): File in which to store output wav files
        dry_run (bool): perform dry run only
    """
    # dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dirname = f"{CLARITY_ROOT}/projects/GHA"
    logging.info(f"Processing {outfile_name} with listener {self.listener}")
    audiogram = GHA.audiogram(self.listener)
    logging.info(f"Audiogram severity is {audiogram.severity}")
    audiogram = audiogram.select_subset_of_cfs(self.audf)

    # Get gain table with noisegate correction
    gaintable = GHA.get_gaintable(
        audiogram,
        self.noisegatelevels,
        self.noisegateslope,
        self.cr_level,
        self.max_output_level,
    )
    formatted_sGt = GHA.format_gaintable(gaintable, noisegate_corr=True)

    # cfg_template = f"{dirname}/cfg_files/{self.cfg_file}_template.cfg"
    cfg_template = CFG_TEMPLATE
    # Merge CH1 and CH3 files. This is the baseline configuration.
    # CH2 is ignored.
    fd_merged, merged_filename = tempfile.mkstemp(
        prefix="clarity-merged-", suffix=".wav"
    )
    # Only need file name; must immediately close the unused file handle.
    os.close(fd_merged)

    # replaces create_HA_inputs
    create_HA_inputs(infile_names, merged_filename)

    # Create the openMHA config file from the template
    fd_cfg, cfg_filename = tempfile.mkstemp(
        prefix="clarity-openmha-", suffix=".cfg"
    )
    # Again, only need file name; must immediately close the unused file handle.
    os.close(fd_cfg)

    with open(cfg_filename, "w") as f:
        f.write(
            GHA.create_configured_cfgfile(
                merged_filename,
                outfile_name,
                formatted_sGt,
                cfg_template,
                self.ahr,
            )
        )

    # Process file using configured cfg file
    # Suppressing OpenMHA output with -q - comment out when testing
    # Append log of OpenMHA commands to /cfg_files/logfile
    subprocess.run(
        [
            "mha",
            "-q",
            "--log=logfile.txt",
            f"?read:{cfg_filename}",
            "cmd=start",
            "cmd=stop",
            "cmd=quit",
        ]
    )

    # Delete temporary files.
    # import ipdb; ipdb.set_trace()

    os.remove(merged_filename)
    os.remove(cfg_filename)

    # Check output signal has energy in every channel
    sig = ccs.read_signal(outfile_name)

    if len(np.shape(sig)) == 1:
        sig = np.expand_dims(sig, axis=1)

    if not np.all(np.sum(abs(sig), axis=0)):
        raise ValueError(f"Channel empty.")
    # import ipdb; ipdb.set_trace()
    # ccs.write_signal(outfile_name, sig, CONFIG.fs, floating_point=True)

    logging.info("customn OpenMHA processing complete")

# Monkey patch the process files method so it will load the clean audio
HearingAid.process_files = process_files
# # skips the differential processing
CONFIG.cfg_file = "modified_prerelease_combination4_smooth"


#############################################################################
# doing what run_HA_processing.py does
from scipy.signal import unit_impulse

# from clarity_core.signal import read_signal, write_signal, pad
sys.path.append("../projects/MSBG")
import MSBG
# to get the hearing loss to work change the working directory
os.chdir(f"{CLARITY_ROOT}/projects/MSBG")

def listen(signal, ears):
    outputs = [
        ear.process(
            signal[:, i],
            add_calibration=CONFIG.calib,
        )
        for i, ear in enumerate(ears)
    ]

    # Fix length difference if no smearing on one of two ears
    if len(outputs[0][0]) != len(outputs[1][0]):
        diff = len(outputs[0][0]) - len(outputs[1][0])
        if diff > 0:
            outputs[1][0] = np.flipud(ccs.pad(np.flipud(outputs[1][0]), len(outputs[0][0])))
        else:
            outputs[0][0] = np.flipud(ccs.pad(np.flipud(outputs[0][0]), len(outputs[1][0])))

    return np.squeeze(outputs).T


def run_HL_processing(scene, listener, input_path, output_path, fs):
    """Run baseline HL processing.
    Applies the MSBG model of hearing loss.
    Args:
        scene (dict): dictionary defining the scene to be generated
        listener (dict): dictionary containing listener data
        input_path (str): path to the input data
        output_path (str): path to the output data
        fs (float): sampling rate
    """
    logging.debug(f"Running HL processing: Listener {listener['name']}")
    logging.debug("Listener data")
    logging.debug(listener["name"])

    # Get audiogram centre frequencies
    cfs = np.array(listener["audiogram_cfs"])

    # Read HA output and mixed signals
    signal = ccs.read_signal(
        f"{input_path}/{scene['scene']}_{listener['name']}_HA-output.wav"
    )

    mixture_signal = ccs.read_signal(f"{input_path}/{scene['scene']}_mixed_CH0.wav")

    # Create discrete delta function (DDF) signal for time alignment
    ddf_signal = np.zeros((np.shape(signal)))
    ddf_signal[:, 0] = unit_impulse(len(signal), int(fs / 2))
    ddf_signal[:, 1] = unit_impulse(len(signal), int(fs / 2))

    # Get flat-0dB ear audiograms
    flat0dB_audiogram = MSBG.Audiogram(cfs=cfs, levels=np.zeros((np.shape(cfs))))
    flat0dB_ear = MSBG.Ear(audiogram=flat0dB_audiogram, src_pos="ff")

    # For flat-0dB audiograms, process the signal with each ear in the list of ears
    flat0dB_HL_outputs = listen(signal, [flat0dB_ear, flat0dB_ear])

    # Get listener audiograms and build a pair of ears
    audiogram_left = np.array(listener["audiogram_levels_l"])
    left_audiogram = MSBG.Audiogram(cfs=cfs, levels=audiogram_left)
    audiogram_right = np.array(listener["audiogram_levels_r"])
    right_audiogram = MSBG.Audiogram(cfs=cfs, levels=audiogram_right)
    audiograms = [left_audiogram, right_audiogram]
    ears = [MSBG.Ear(audiogram=audiogram, src_pos="ff") for audiogram in audiograms]

    # Process the HA output signal, the raw mixed signal, and the ddf signal
    outputs = listen(signal, ears)
    mixture_outputs = listen(mixture_signal, ears)
    ddf_outputs = listen(ddf_signal, ears)

    # Write the outputs
    outfile_stem = f"{output_path}/{scene['scene']}_{listener['name']}"
    signals_to_write = [
        (
            flat0dB_HL_outputs,
            f"{output_path}/{scene['scene']}_flat0dB_HL-output.wav",
        ),
        (outputs, f"{outfile_stem}_HL-output.wav"),
        (ddf_outputs, f"{outfile_stem}_HLddf-output.wav"),
        (mixture_outputs, f"{outfile_stem}_HL-mixoutput.wav"),
    ]
    for signal, filename in signals_to_write:
        ccs.write_signal(filename, signal, CONFIG.fs, floating_point=True)


#############################################################################
# doing what run_HA_processing.py does

# import sys
# import argparse
# import json
# import numpy as np
# import logging
# from tqdm import tqdm
#
# from clarity_core.config import CONFIG
# from clarity_core.signal import read_signal, find_delay_impulse

sys.path.append("../projects/MBSTOI")
from MBSTOI import mbstoi


def calculate_SI(
    scene,
    listener,
    clean_input_path,
    processed_input_path,
    fs,
    gridcoarseness=1,
    dry_run=False,
):
    """Run baseline speech intelligibility (SI) algorithm. MBSTOI
    requires time alignment of input signals. Here we correct for
    broadband delay introduced by the MSBG hearing loss model.
    Hearing aids also introduce a small delay, but this depends on
    the exact implementation. See projects/MBSTOI/README.md.
    Outputs can be found in text file sii.txt in /scenes folder.
    Args:
        scene (dict): dictionary defining the scene to be generated
        listener (dict): listener
        clean_input_path (str): path to the clean speech input data
        processed_input_path (str): path to the processed input data
        fs (float): sampling rate
        gridcoarseness (int): MBSTOI EC search grid coarseness (default: 1)
        dry_run (bool, optional): run in dry_run mode (default: False)
    """
    logging.info(
        f"Running SI calculation: scene {scene['scene']}, listener {listener['name']}"
    )

    # Get non-reverberant clean signal
    clean = ccs.read_signal(f"{clean_input_path}/{scene['scene']}_target_anechoic.wav")

    # Get signal processed by HL and HA models
    proc = ccs.read_signal(
        f"{processed_input_path}/{scene['scene']}_{listener['name']}_HL-output.wav",
    )

    # Calculate channel-specific unit impulse delay due to HL model and audiograms
    ddf = ccs.read_signal(
        f"{processed_input_path}/{scene['scene']}_{listener['name']}_HLddf-output.wav",
    )
    delay = ccs.find_delay_impulse(ddf, initial_value=int(CONFIG.fs / 2))

    if delay[0] != delay[1]:
        logging.info(f"Difference in delay of {delay[0] - delay[1]}.")

    maxdelay = int(np.max(delay))

    # Allow for value lower than 1000 samples in case of unimpaired hearing
    if maxdelay > 2000:
        logging.error(f"Error in delay calculation for signal time-alignment.")

    # # For baseline software test signals, MBSTOI index tends to be higher when
    # # correcting for ddf delay + length difference.
    # diff = len(proc) - len(clean)
    # if diff < 0:
    #     logging.error("Check signal length!")
    #     diff = 0
    # else:
    #     logging.info(
    #         f"Correcting for delay + difference in signal lengths where delay = {delay} and length diff is {diff} samples."
    #     )

    # delay[0] += diff
    # delay[1] += diff

    # Correct for delays by padding clean signals
    cleanpad = np.zeros((len(clean) + maxdelay, 2))
    procpad = np.zeros((len(clean) + maxdelay, 2))

    if len(procpad) < len(proc):
        raise ValueError(f"Padded processed signal is too short.")

    cleanpad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]
    cleanpad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]
    procpad[: len(proc)] = proc

    # Calculate intelligibility
    if dry_run:
        return
    else:
        sii = mbstoi(
            cleanpad[:, 0],
            cleanpad[:, 1],
            procpad[:, 0],
            procpad[:, 1],
            gridcoarseness=gridcoarseness,
        )

        logging.info(f"{sii:3.4f}")
        return sii


if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dry_run", action="store_true", help="perform dry run only")
#     parser.add_argument(
#         "--num_channels",
#         nargs="?",
#         type=int,
#         default=1,
#         help="number of HA channels [default: 1]",
#     )
#     parser.add_argument("scene_list_filename", help="json file containing scene data")
#     parser.add_argument("listener_filename", help="json file containing listener data")
#     parser.add_argument(
#         "scenes_listeners_filename",
#         help="json file containing scenes to listeners mapping",
#     )

#     parser.add_argument("input_path", help="input file names")
#     parser.add_argument("output_path", help="path to output data")
#     args = parser.parse_args()

#     channels = args.num_channels
    ############################################################################

    # from `run_HA_processing.py`
    scene_list = json.load(open(scene_list_filename, "r"))
    listeners = json.load(open(listener_filename, "r"))
    scenes_listeners = json.load(open(scenes_listeners_filename, "r"))
    hearing_aid = HearingAid(fs=CONFIG.fs, channels=3)
    for scene_n, scene in enumerate(scene_list):
        for listener_name in scenes_listeners[scene["scene"]]:
            infile_names=[
                f"{input_path}/{scene['scene']}_mixed_CH{ch}.wav"
                for ch in range(1, channels + 1)
            ]
            if all([Path(file).exists() for file in infile_names]):
                listener = listeners[listener_name]
                hearing_aid.set_listener(listener)
                ##########
                # baseline model
                # hearing_aid.process_files(
                #     infile_names=[
                #         f"{input_path}/{scene['scene']}_mixed_CH{ch}.wav"
                #         for ch in range(1, channels + 1)
                #     ],
                #     outfile_name=(
                #         f"{output_path}/{scene['scene']}_{listener['name']}_HA-output.wav"
                #     ),
                # )

                ##########
                #  cleaned inputs as input to hearing aids
                # hearing_aid.process_files(
                #     infile_names=[
                #         f"{input_path}/{scene['scene']}_cleaned_signal_441k.wav"
                #     ],
                #     outfile_name=(
                #         f"{output_path}/{scene['scene']}_{listener['name']}_HA-output.wav"
                #     ),
                # )

                ######################################################################
                # no hearing aid apart from what the unet does
                # shutil.copy(f"{input_path}/{scene['scene']}_cleaned_signal_441k.wav", f"{output_path}/{scene['scene']}_{listener['name']}_HA-output.wav")

                ######################################################################
                # custom hearing aid
                # window_len = int(10/16000 * 44100)
                window_len = 11
                audio, fs_hearing_aid = sf.read(f"{input_path}/{scene['scene']}_cleaned_signal_16k.wav")
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

                # load the origianl channel 1 to find out the length:
                original_ch1, fs = sf.read(Path(input_path) / f"{scene['scene']}_mixed_CH1.wav")
                original_samples = original_ch1.shape[0]
                fil_audio_compressed_44k = librosa.resample(fil_audio_compressed.T, fs_hearing_aid, 44100)
                fil_audio_compressed_44k = fil_audio_compressed_44k[:,0:original_samples].T
                sf.write(f"{output_path}/{scene['scene']}_{listener['name']}_HA-output.wav", fil_audio_compressed_44k, 44100, subtype="FLOAT")
                # check the output is syncronised with the input (CH1, left ear)
                corr = signal.correlate(original_ch1[:, 0], fil_audio_compressed_44k[: ,0], mode='full', method='auto')
                in2_len = fil_audio_compressed_44k.shape[0]
                in1_len = original_ch1.shape[0]
                lags = np.arange(-in2_len + 1, in1_len)
                delay = lags[np.argmax(np.abs(corr))]
                assert delay == 0, "Signal is not properly synced with input"

                ######################################################################
                # Run the HL processing
                # $PYTHON_BIN "$CLARITY_ROOT"/scripts/run_HL_processing.py  "$CLARITY_DATA"/metadata/scenes."$dataset".json "$CLARITY_DATA"/metadata/listeners.json "$CLARITY_DATA"/metadata/scenes_listeners."$dataset".json "$CLARITY_DATA"/"$dataset"/scenes "$CLARITY_DATA"/"$dataset"/scenes
                run_HL_processing(
                    scene,
                    listener,
                    input_path,
                    output_path,
                    CONFIG.fs,
                )

                ######################################################################
                # SI speech intelligbility
                # e.g. python calculate_SI.py "../data/scenes/train.json" "../data/listeners.json" "../data/scenes_listeners.json" "../data/output/train" "../data/output/train"
                sii = calculate_SI(
                    scene,
                    listener,
                    input_path,
                    output_path,
                    CONFIG.fs,
                    dry_run=False,
                )
                print(f"Scene {scene['scene']} listener {listener['name']} sii {round(sii,4)}\n")
                # output_file.write(
                #     f"Scene {scene['scene']} listener {listener['name']} sii {round(sii,4)}\n"
                # )

    # Run the intelligibility model
    # $PYTHON_BIN "$CLARITY_ROOT"/scripts/calculate_SI.py "$CLARITY_DATA"/metadata/scenes."$dataset".json "$CLARITY_DATA"/metadata/listeners.json "$CLARITY_DATA"/metadata/scenes_listeners."$dataset".json "$CLARITY_DATA"/"$dataset"/scenes "$CLARITY_DATA"/"$dataset"/scenes
