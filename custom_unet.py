# https://github.com/karolzak/keras-unet
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    UpSampling2D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
)
def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

#
# def custom_unet(
#     input_shape,
#     output_channels=1,
#     activation="relu",
#     use_batch_norm=True,
#     upsample_mode="deconv",  # 'deconv' or 'simple'
#     dropout=0.3,
#     dropout_change_per_layer=0.0,
#     dropout_type="spatial",
#     use_dropout_on_upsampling=False,
#     use_attention=False,
#     filters=16,
#     num_layers=4,
#     mag_activation="sigmoid",
#     phase_activation=None
# ):  # 'sigmoid' or 'softmax'
#
#     """
#     Customisable UNet architecture (Ronneberger et al. 2015 [1]).
#     Arguments:
#     input_shape: 3D Tensor of shape (x, y, num_channels)
#     output_channels (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
#     activation (str): A keras.activations.Activation to use. ReLu by default.
#     use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
#     upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
#     dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
#     dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
#     dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
#     use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
#     use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
#     filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
#     num_layers (int): Number of total layers in the encoder not including the bottleneck layer
#     output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
#     Returns:
#     model (keras.models.Model): The built U-Net
#     Raises:
#     ValueError: If dropout_type is not one of "spatial" or "standard"
#     [1]: https://arxiv.org/abs/1505.04597
#     [2]: https://arxiv.org/pdf/1411.4280.pdf
#     [3]: https://arxiv.org/abs/1804.03999
#     """
#
#     if upsample_mode == "deconv":
#         upsample = upsample_conv
#     else:
#         upsample = upsample_simple
#
#     # Build U-Net model
#     inputs = Input(input_shape)
#     x = inputs
#
#     # pad spectrogram dimensions to the next power of twov
#     spectrogram_shape = x.shape[1:3]
#     pad_bins = next_power_of_two(spectrogram_shape[0]) - spectrogram_shape[0]
#     pad_frames = next_power_of_two(spectrogram_shape[1]) - spectrogram_shape[1]
#     x = tf.pad(x, [[0, 0], [0, pad_bins] , [0, pad_frames], [0, 0]])
#
#     down_layers = []
#     for l in range(num_layers):
#         x = conv2d_block(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#         down_layers.append(x)
#         x = MaxPooling2D((2, 2))(x)
#         dropout += dropout_change_per_layer
#         filters = filters * 2  # double the number of filters with each layer
#
#     x = conv2d_block(
#         inputs=x,
#         filters=filters,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         activation=activation,
#     )
#
#     if not use_dropout_on_upsampling:
#         dropout = 0.0
#         dropout_change_per_layer = 0.0
#
#     for conv in reversed(down_layers):
#         filters //= 2  # decreasing number of filters with each layer
#         dropout -= dropout_change_per_layer
#         x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
#         if use_attention:
#             x = attention_concat(conv_below=x, skip_connection=conv)
#         else:
#             x = concatenate([x, conv])
#         x = conv2d_block(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#
#     mag = Conv2D(1, (1, 1), activation=mag_activation)(x)
#     mag = tf.slice(mag, [0,0,0,0], [-1,spectrogram_shape[0],spectrogram_shape[1],-1])
#     if phase_activation:
#         phase = Conv2D(1, (1, 1), activation=phase_activation)(x)
#         phase = tf.slice(mag, [0,0,0,0], [-1,spectrogram_shape[0],spectrogram_shape[1],-1])
#         return Model(inputs=[inputs], outputs=[mag, phase])
#
#     return Model(inputs=[inputs], outputs=[mag])
#


def custom_unet(
    input_shape,
    input_type="mag",
    output_channels=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    mag_activation="sigmoid",
    phase_activation=None
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    output_channels (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999
    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    mag_input = Input(input_shape)
    # pad spectrogram dimensions to the next power of twov
    spectrogram_shape = input_shape
    pad_bins = next_power_of_two(spectrogram_shape[0]) - spectrogram_shape[0]
    pad_frames = next_power_of_two(spectrogram_shape[1]) - spectrogram_shape[1]
    x = tf.pad(mag_input, [[0, 0], [0, pad_bins] , [0, pad_frames], [0, 0]])

    if input_type == "mag":
        inputs = mag_input
    elif input_type == "mag_phase":
        phase_input = Input(input_shape)
        phase_input = tf.pad(phase_input, [[0, 0], [0, pad_bins] , [0, pad_frames], [0, 0]])
        x = concatenate([mag_input, phase_input])

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    mag = Conv2D(1, (1, 1), activation=mag_activation)(x)
    mag = tf.slice(mag, [0,0,0,0], [-1,spectrogram_shape[0],spectrogram_shape[1],-1])
    if phase_activation:
        phase = Conv2D(1, (1, 1), activation=phase_activation)(x)
        mag = tf.slice(phase, [0,0,0,0], [-1,spectrogram_shape[0],spectrogram_shape[1],-1])
        return Model(inputs=[inputs], outputs=[mag, phase])

    return Model(inputs=[inputs], outputs=[mag])

def next_power_of_two(x):
    exponent = np.ceil(np.log2(x)/np.log2(2))
    return int(2**exponent)


# def custom_custom_unet(
#     input_shape,
#     output_channels=1,
#     activation="relu",
#     use_batch_norm=True,
#     upsample_mode="deconv",  # 'deconv' or 'simple'
#     dropout=0.3,
#     dropout_change_per_layer=0.0,
#     dropout_type="spatial",
#     use_dropout_on_upsampling=False,
#     use_attention=False,
#     filters=16,
#     num_layers=4,
#     mag_activation="sigmoid",
#     phase_activation="sigmoid",
#     frame_step=80,
#     frame_size=1024,
#     fft_n=1024,
#     trainable_spec_layers=False
# ):
#     """
#     Customisable UNet architecture (Ronneberger et al. 2015 [1]).
#     Arguments:
#     input_shape: 3D Tensor of shape (x, y, num_channels)
#     output_channels (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
#     activation (str): A keras.activations.Activation to use. ReLu by default.
#     use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
#     upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
#     dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
#     dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
#     dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
#     use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
#     use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
#     filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
#     num_layers (int): Number of total layers in the encoder not including the bottleneck layer
#     output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
#     Returns:
#     model (keras.models.Model): The built U-Net
#     Raises:
#     ValueError: If dropout_type is not one of "spatial" or "standard"
#     [1]: https://arxiv.org/abs/1505.04597
#     [2]: https://arxiv.org/pdf/1411.4280.pdf
#     [3]: https://arxiv.org/abs/1804.03999
#     """
#     if upsample_mode == "deconv":
#         upsample = upsample_conv
#     else:
#         upsample = upsample_simple
#
#     # Build U-Net model
#     inputs = Input(input_shape)
#
#     stft_layer = Stft(
#         frame_size=frame_size,
#         fft_n=frame_size,
#         frame_step=frame_step,
#         window_name="hann_window",
#         pad_end=True
#     )
#     stft_layer.trainable = trainable_spec_layers
#     x = stft_layer(inputs)
#     mag_in = tf.math.sqrt(tf.square(x[...,0]) + tf.square(x[...,1]))
#     phase_in = tf.math.atan2(x[...,1], x[...,0])
#     x = tf.keras.layers.Concatenate(axis=-1)([mag_in, phase_in])
#     spectrogram_shape = x.shape[1:3]
#     print(x)
#
#     # pad spectrogram dimensions to the next power of two
#     pad_bins = next_power_of_two(spectrogram_shape[0]) - spectrogram_shape[0]
#     pad_frames = next_power_of_two(spectrogram_shape[1]) - spectrogram_shape[1]
#     x = tf.pad(x, [[0, 0], [0, pad_bins] , [0, pad_frames], [0, 0]])
#
#     # change to bins, frames, channels
#     x = tf.transpose(x, [0, 2, 1, 3])
#     print(x)
#
#     down_layers = []
#     for l in range(num_layers):
#         x = conv2d_block(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#         down_layers.append(x)
#         x = MaxPooling2D((2, 2))(x)
#         dropout += dropout_change_per_layer
#         filters = filters * 2  # double the number of filters with each layer
#
#     x = conv2d_block(
#         inputs=x,
#         filters=filters,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         activation=activation,
#     )
#
#     if not use_dropout_on_upsampling:
#         dropout = 0.0
#         dropout_change_per_layer = 0.0
#
#     for conv in reversed(down_layers):
#         filters //= 2  # decreasing number of filters with each layer
#         dropout -= dropout_change_per_layer
#         x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
#         if use_attention:
#             x = attention_concat(conv_below=x, skip_connection=conv)
#         else:
#             x = concatenate([x, conv])
#         x = conv2d_block(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#
#     spectrogram_output = Conv2D(output_channels, (1, 1), activation=output_activation)(x)
#     model = Model(inputs=[inputs], outputs=[spectrogram_output])
#     # remove spectrogram padding
#     spectrogram_output = tf.slice(spectrogram_output, [0,0,0,0], [-1,spectrogram_shape[0],spectrogram_shape[1],-1])
#     model = Model(inputs=[inputs], outputs=[spectrogram_output])
#     return model
#
#     # create an extra penultimate dim, which is the audio channels dimensions, last dim is real / imag
#     stft = tf.expand_dims(spectrogram_output, -2)
#     istft_layer = Istft(
#         frame_size=frame_size,
#         fft_n=fft_n,
#         frame_step=frame_step,
#         window_name="hann_window"
#     )
#     istft_layer.trainable = trainable_spec_layers
#     waveform_output = istft_layer(stft)
#     waveform_output = tf.slice(waveform_output, [0,0,0], [-1, input_shape[0], -1])
#     waveform_output = tf.keras.layers.Lambda(lambda x: x, name="waveform")(waveform_output)
#     mag_out = tf.keras.layers.Lambda(lambda x: x, name="spectrogram")(stft)
#
#     model = Model(inputs=[inputs], outputs=[waveform_output, spectrogram_output])
#
#     return model
