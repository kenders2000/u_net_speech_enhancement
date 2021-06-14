"""
This script converts the output of the u-net scene cleaning to the format
required by the hearing aid processor.
"""
import librosa
import soundfile as sf
from pathlib import Path
import os
import tqdm

# clean_path = Path("/home/paulkendrick/spectrogram_based_models/example_data/clarity_CEC1_data/clarity_data/dev/cleaned_audio")
# data_path = Path("/home/paulkendrick/spectrogram_based_models/example_data/clarity_CEC1_data/clarity_data/dev/scenes")

def main():
    ap = ArgumentParser()
    ap.add_argument(
        "-p",
        type=str,
        dest="clean_path",
        help="Path to the cleaned audio (output of u-net).",
        default="/home/paulkendrick/spectrogram_based_models/example_data/clarity_CEC1_data/clarity_data/dev/cleaned_audio",
    )
    ap.add_argument(
        "-d",
        type=str,
        dest="data_path",
        help="Path to the scenes (where _mixed_CH1.wav are).",
        default="/home/paulkendrick/spectrogram_based_models/example_data/clarity_CEC1_data/clarity_data/dev/scenes",
    )
    ap.add_argument(
        "-o",
        type=str,
        dest="output_path",
        help="Path save the converted files.",
        default="/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/spectrogram_models/cleaned_scenes",
    )
    args = ap.parse_args()
    clean_path = Path(args.clean_path)
    data_path = Path(args.clean_path)
    output_path = Path(args.output_path)
    # traverse whole directory
    for root, dirs, files in os.walk(clean_path):
        # select file name
        for file in files:
            # check the extension of files
            if file.endswith('16k.wav'):
                reconstructed_audio_full, fs = sf.read(clean_path / file)
                scene = Path(file).stem.split("_")[0]

                output_filename_44k = f"{scene}_cleaned_signal_441k.wav"
                reconstructed_audio_full_44k = librosa.resample(reconstructed_audio_full.T, fs, 44100)

                # load the origianl channel 1 to find out the length:
                original_ch1, fs = sf.read(data_path / f"{scene}_mixed_CH1.wav")
                original_samples = original_ch1.shape[0]
                reconstructed_audio_full_44k = reconstructed_audio_full_44k[:,0:original_samples]
                output_filename_44k = f"{scene}_cleaned_signal_441k.wav"
                sf.write(clean_path / output_filename_44k, reconstructed_audio_full_44k.T, 44100, subtype="FLOAT")

    # legacy conversion:
    # I saved the cleaned output as {scene}_{listener}_HA-output.wav as 16 kHz 16 bit wavs
    # this is the output of the u-net.
    # they are to be all upsampled to 44.1 kHz and save as float wavs named:
    # {scene}_cleaned_signal_441k.wav, this cleaning is listener independnat, and will
    # then be passed into a listening aid algorithm.

    clean_path = Path("/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/spectrogram_models/reconstructed_audio/eval")
    data_path = Path("/home/kenders/greenhdd/clarity_challenge/data/clarity_CEC1_data/clarity_data/eval/scenes/")
    output_path = Path("/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/spectrogram_models/cleaned_scenes")
    # traverse whole directory
    for root, dirs, files in os.walk(clean_path):
        # select file name
        for file in tqdm.tqdm(files):
            # check the extension of files
            if file.endswith('HA-output.wav'):
                reconstructed_audio_full, fs = sf.read(clean_path / file)
                scene = Path(file).stem.split("_")[0]
                output_filename = f"{scene}_cleaned_signal_16k.wav"
                sf.write(output_path / output_filename, reconstructed_audio_full, 16000, subtype="FLOAT")
                # reconstructed_audio_full_44k = librosa.resample(reconstructed_audio_full.T, fs, 44100)
                #
                # # load the origianl channel 1 to find out the length:
                # # this is because I zeropadded all signals to the same length.
                # original_ch1, fs = sf.read(data_path / f"{scene}_mixed_CH1.wav")
                # original_samples = original_ch1.shape[0]
                # reconstructed_audio_full_44k = reconstructed_audio_full_44k[:, 0:original_samples]
                # output_filename_44k = f"{scene}_cleaned_signal_441k.wav"
                #
                # sf.write(output_path / output_filename_44k, reconstructed_audio_full_44k.T, 44100, subtype="FLOAT")
