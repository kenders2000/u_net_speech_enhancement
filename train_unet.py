from argparse import ArgumentParser
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


def chunk_list(input_list, chunk_size):
    """Convert list into a list of lists."""
    return [
        input_list[chunk : chunk + chunk_size]
        for chunk in range(0, len(input_list), chunk_size)
    ]


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "-p",
        type=str,
        dest="path_to_model_starting_point",
        help="Path a starting point for the model.",
        default=None
        # default="/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/models/1_2.121e-07/",
    )
    args = ap.parse_args()
    eps = 1e-9
    fs = 16000

    spec_frame_size = 1024 # odd fft ensures even rfft
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
        n_proc=1,
        subset_size_ratio=0.1
    )

    x_spec, y_spec = data_loader[0]
    frames = x_spec.shape[1]
    bins = x_spec.shape[2]
    channels = x_spec.shape[3]

    model = custom_unet(
        input_shape=(frames, bins, 6),
        input_type="mag",
        output_channels=1,
        activation="relu",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        dropout=0.3,
        dropout_change_per_layer=0.0,
        dropout_type="spatial",
        use_dropout_on_upsampling=False,
        use_attention=True,
        filters=16,
        num_layers=4,
        mag_activation="relu",
        phase_activation=None,
    )
    starting_epoch = 0
    if args.path_to_model_starting_point:
        print("Loading model")
        model = tf.keras.models.load_model(args.path_to_model_starting_point)
        starting_epoch = int(Path(args.path_to_model_starting_point).stem.split("_")[0]) + 1

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanAbsoluteError(
        reduction="auto", name="mean_absolute_error"
    )


    checkpoint_filepath = Path('/home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement/models')
    if not checkpoint_filepath.exists():
        os.makedirs(checkpoint_filepath)

    epochs = 20
    sub_batch_size = 3

    enq = tf.keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=False)
    enq.start(workers=1)
    gen = enq.get()


    for epoch in range(starting_epoch, epochs):
        print(f"epoch {epoch} \n")
        progbar = tf.keras.utils.Progbar(len(data_loader))

        # model.save(checkpoint_filepath / f"{epoch}_start")
        # Iterate over the batches of the dataset.
        for batch_n in range(len(data_loader)):
            x_spec, y_spec = next(gen)
            # print(f"epoch {epoch} batch {batch_n} of {len(data_loader)}")
            # the batches are large and need to split into smaller chunks for feeding into the network
            x_spec, y_spec = sklearn.utils.shuffle(x_spec, y_spec)
            x_spec_chunked, y_spec_chunked = chunk_list(x_spec, sub_batch_size), chunk_list(y_spec, sub_batch_size)

            for sub_batch, (x_batch_train, y_batch_train) in enumerate(zip(x_spec_chunked, y_spec_chunked)):
                with tf.device('/gpu:0'):
                    with tf.GradientTape() as tape:
                        logits = model(x_batch_train, training=True)  # Logits for this minibatch
                        # Compute the loss value for this minibatch.
                        loss_value = loss_fn(y_batch_train, logits)

                    # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
                    grads = tape.gradient(loss_value, model.trainable_weights)

                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    opt.apply_gradients(zip(grads, model.trainable_weights))
            progbar.add(1, values=[("loss", float(loss_value))])
        model.save(checkpoint_filepath / f"{epoch}_{float(loss_value):.4}")

if __name__ ==  "__main__":
    main()
