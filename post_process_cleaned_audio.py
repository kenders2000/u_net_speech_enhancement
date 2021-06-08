# $PYTHON_BIN "$CLARITY_ROOT"/scripts/run_HA_processing.py  --num_channels "$num_channels" "$CLARITY_DATA"/metadata/scenes."$dataset".json "$CLARITY_DATA"/metadata/listeners.json "$CLARITY_DATA"/metadata/scenes_listeners."$dataset".json "$CLARITY_DATA"/"$dataset"/scenes "$CLARITY_DATA"/"$dataset"/scenes
## Set these paths
CLARITY_ROOT = "/home/paulkendrick/clarity_CEC1"
CLARITY_DATA = "/home/paulkendrick/clarity_CEC1/data/clarity_data"
CFG_TEMPLATE = "/home/paulkendrick/spectrum_models/modified_prerelease_combination4_smooth_template.cfg"


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
# skips the differential processing
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
    hearing_aid = HearingAid(fs=CONFIG.fs, channels=channels)
    for scene_n, scene in tqdm(enumerate(scene_list)):
        for listener_name in scenes_listeners[scene["scene"]]:
            infile_names=[
                f"{input_path}/{scene['scene']}_mixed_CH{ch}.wav"
                for ch in range(1, channels + 1)
            ]
            print(scene_n, listener_name)
            if all([Path(file).exists() for file in infile_names]):
                listener = listeners[listener_name]
                hearing_aid.set_listener(listener)
                hearing_aid.process_files(
                    infile_names=[
                        f"{input_path}/{scene['scene']}_mixed_CH{ch}.wav"
                        for ch in range(1, channels + 1)
                    ],
                    outfile_name=(
                        f"{output_path}/{scene['scene']}_{listener['name']}_HA-output.wav"
                    ),
                )

                # Run the HL processing
                # $PYTHON_BIN "$CLARITY_ROOT"/scripts/run_HL_processing.py  "$CLARITY_DATA"/metadata/scenes."$dataset".json "$CLARITY_DATA"/metadata/listeners.json "$CLARITY_DATA"/metadata/scenes_listeners."$dataset".json "$CLARITY_DATA"/"$dataset"/scenes "$CLARITY_DATA"/"$dataset"/scenes
                run_HL_processing(
                    scene,
                    listeners[listener_name],
                    input_path,
                    output_path,
                    CONFIG.fs,
                )

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
