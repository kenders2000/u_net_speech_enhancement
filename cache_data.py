from dataloading import ClarityAudioDataloaderSequenceAudio, ClarityAudioDataloaderSequenceSpectrograms
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
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(list_of_ints):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def _process_shard(x_spec, y_spec, shard_path, options):
    """Serialise and save data for a single shard.

    Args:
        x_spec (numpy array): input batch
        y_spec (numpy array): target batch
        shard_path(pathlib Path): the full path (including filename) to
            write the shard to
        options(str): TFRecordWriter compression options e.g. `ZLIB`
    """
    with tf.io.TFRecordWriter(shard_path.as_posix(), options=options) as out:
        record = {
            "x_spec": _bytes_feature(
                [np.reshape(x_spec, -1).astype(np.float32).tobytes()]
            ),
            "x_spec_shape": _int_feature(x_spec.shape),
            "y_spec": _bytes_feature(
                [np.reshape(y_spec, -1).astype(np.float32).tobytes()]
            ),
            "y_spec_shape": _int_feature(y_spec.shape),
        }  # Create a Features message using tf.train.Example.
        example = tf.train.Example(
            features=tf.train.Features(feature=record)
        ).SerializeToString()
        out.write(example)

def chunk_list(input_list, chunk_size):
    """Convert list into a list of lists."""
    return [
        input_list[chunk : chunk + chunk_size]
        for chunk in range(0, len(input_list), chunk_size)
    ]


def main():
    eps = 1e-9
    fs = 16000

    spec_frame_size = 512 # odd fft ensures even rfft
    spec_frame_step = spec_frame_size // 4 #int(fs * 5e-3) // 2

    channels_in = 6
    lookahead_frame_size = int(6.0*fs)
    lookahead_frame_step = int(np.floor(5e-3 * fs))
    batch_size = 2

    data_loader = ClarityAudioDataloaderSequenceSpectrograms(
        spec_frame_step=spec_frame_step,
        spec_frame_size=spec_frame_size,
        frame_step=lookahead_frame_step,
        frame_size=lookahead_frame_size,
        new_sample_rate=fs,
        batch_size=batch_size,
        target_length=6.0*fs,
        verbose=0,
        return_type="abs",
        n_proc=16,
        subset_size_ratio=0.1
    )


    epochs = 20

    enq = tf.keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=False)
    enq.start(workers=1)
    gen = enq.get()

    progbar = tf.keras.utils.Progbar(len(data_loader))
    options="ZLIB"
    cache_dir = Path("/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/spectrogram_models/cache/")
    dataset_name = "train"
    for epoch in range(epochs):
        print(f"epoch {epoch} \n")
        # model.save(checkpoint_filepath / f"{epoch}_start")
        # Iterate over the batches of the dataset.
        for batch_n in range(len(data_loader)):
            shard_path = cache_dir / dataset_name / "{}-{:03d}-{}.npz".format(
                        dataset_name, batch_n, batch_size)
            x_spec, y_spec = next(gen)
            np.savez(shard_path, x_spec=x_spec, y_spec=y_spec)
            # _process_shard(x_spec, y_spec, shard_path, options)
            progbar.add(1)

if __name__ ==  "__main__":
    main()
