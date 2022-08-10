import math
import numpy as np
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import layers, Model


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

BlockArgs = namedtuple('BlockArgs',
                       ('kernel_size', 'input_filters', 'output_filters', 'pool_size',
                        'dilation', 'strides', 'expand_ratio', 'se_ratio',
                        'id_skip', 'dec_skip'),
                       defaults=(3, None, 8, 1, 1, 1, 6, 0.25, True, False))

OutputArgs = namedtuple('OutputArgs',(
    #'kernel_size', 'filters', 'pool_size', 'dilation', 'strides', 'se_ratio'))
    'output_name', 'n_classes', 'samples_per_segment', 'segment_ksize',
    'segment_activation', 'dense_ksize', 'dense_activation'),
    defaults=(None, 2, None, 1, 'softmax', 1, 'tanh'))


DEFAULT_BLOCK_ARGS = [
    BlockArgs(kernel_size=5, output_filters=32, pool_size=8, dilation=2, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=5, output_filters=48, pool_size=6, dilation=2, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=5, output_filters=64, pool_size=4, dilation=2, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=3, output_filters=96, pool_size=2, dilation=1, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=3, output_filters=128, pool_size=2, dilation=1, strides=1, se_ratio=0.1),
    #BlockArgs(kernel_size=3, input_filters=128, pool_size=2, dilation=1, strides=1, se_ratio=0.1),
    BlockArgs(kernel_size=3, output_filters=256, pool_size=1, dilation=1, strides=1, se_ratio=0.1),
]


DEFAULT_OUTPUT_ARGS = [
    OutputArgs(output_name='hypnogram', n_classes=5, samples_per_segment=None, segment_ksize=1,
               segment_activation='softmax', dense_ksize=1, dense_activation='tanh')
]


def squeeze_and_excite_1d(input_tensor, se_ratio=0.25, channel_axis=-1, activation='relu'):
    if activation == 'swish':
        activation = tf.nn.swish
    input_shape = input_tensor.get_shape().as_list()
    filters = input_shape[channel_axis]
    num_reduced_filters = max(1, int(filters*se_ratio))
    se_shape = (1, filters)
    x = layers.GlobalAvgPool1D()(input_tensor)
    x = layers.Reshape(se_shape)(x)
    x = layers.Conv1D(
        num_reduced_filters, 1,
        activation=activation,
        strides=1,
        padding='same',
        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.Conv1D(
        filters, 1,
        activation='sigmoid',
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding='same')(x)
    
    return layers.multiply([x, input_tensor])


def ASPP_1D(inputs, depth=128, activation=tf.nn.relu, atrous_rates=[6, 12, 18],
            ksizes=[5, 5, 5], conv_cls=layers.Conv1D):
    """Atrous spatial pyramid pooling
    
    https://arxiv.org/pdf/1706.05587.pdf
    https://github.com/rishizek/tensorflow-deeplab-v3/blob/master/deeplab_model.py#L21
    https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/_deeplab.py
    """
    conv_1 = conv_cls(depth, 1, strides=1, padding='same', use_bias=False)(inputs)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = activation(conv_1)
    
    dilated_1 = conv_cls(depth, ksizes[0], dilation_rate=atrous_rates[0],
                                   padding='same', use_bias=False)(inputs)
    dilated_1 = layers.BatchNormalization()(dilated_1)
    dilated_1 = activation(dilated_1)
    
    dilated_2 = conv_cls(depth, ksizes[1], dilation_rate=atrous_rates[1],
                                   padding='same', use_bias=False)(inputs)
    dilated_2 = layers.BatchNormalization()(dilated_2)
    dilated_2 = activation(dilated_2)
    
    dilated_3 = conv_cls(depth, ksizes[2], dilation_rate=atrous_rates[2],
                                   padding='same', use_bias=False)(inputs)
    dilated_3 = layers.BatchNormalization()(dilated_3)
    dilated_3 = activation(dilated_3)
    
    pooled = tf.reduce_mean(inputs, axis=1, keepdims=True)
    pooled = layers.Conv1D(depth, 1, padding='same', use_bias=False)(pooled)
    pooled = layers.BatchNormalization()(pooled)
    pooled = activation(pooled)
    pooled = tf.expand_dims(pooled, axis=2)
    up_shape = tf.convert_to_tensor([tf.shape(inputs)[1], 1])
    pooled = tf.image.resize(pooled, up_shape)
    pooled = tf.squeeze(pooled, axis=2)
    
    concatenated = layers.Concatenate(axis=-1)([conv_1, dilated_1, dilated_2, dilated_3, pooled])
    
    projected = layers.Conv1D(depth, 1, padding='same', use_bias=False)(concatenated)
    projected = layers.BatchNormalization()(projected)
    projected = activation(projected)
    return projected


def ASPP(inputs, depth=128, activation=tf.nn.relu, atrous_rates=[6, 12, 18],
    ksizes=[5, 5, 5], conv_cls=layers.Conv1D):
    """Create a Keras Model from the ASPP_1D function for
    prettier summary.
    """
    inps = layers.Input(shape=inputs.shape[1:], name='aspp_input')
    outputs = ASPP_1D(
        inps, depth=depth, activation=activation,
        atrous_rates=atrous_rates, ksizes=ksizes,
        conv_cls=conv_cls
    )
    aspp_m = tf.keras.Model(inputs=inps, outputs=outputs, name='ASPP')
    return aspp_m(inputs)


@tf.function
def pad_nodes_to_match(node1, node2):
    """Pad to same shape if needed.
    
    This may be necessary when pooling leads to fractionated output shape.
    E.g. if pool_size is 4, and input length is 10, pooling leads to
    10/4 = 2.5. In this case, Keras pooling layer implicitly crops the output to
    length 2. This needs to be taken into account by padding the up path output
    in upsampling part.
    
    NOTE: node1 should be the potentially larger one.
    
    Credit: https://github.com/perslev/U-Time/blob/master/utime/models/utime.py
    """
    s1 = tf.shape(node1)
    s2 = tf.shape(node2)
    if s1[1] != s2[1]:
        diffs = s1[1] - s2[1]
        left_pad = diffs // 2
        right_pad = diffs // 2
        right_pad = right_pad + (diffs % 2)
        pads = tf.convert_to_tensor([[0, 0], [left_pad, right_pad], [0, 0], [0, 0]])
        return tf.pad(node2, pads, 'CONSTANT')
    else:
        return node2


class Conv1DBlock(layers.Layer):
    """An 1D convolutional block of conv+batchnorm+conv+batchnorm+pooling.
    
    Implemented with Conv2D under the hood.
    """
    def __init__(self, block_args,
                 activation='relu',
                 dataformat='channels_last',
                 padding='same',
                 dropout=None,
                 conv_type='conv',
                 **kwargs):
        super().__init__(**kwargs)
        if type(block_args) is not BlockArgs:
            block_args = BlockArgs(**block_args)
        self.block_args = block_args
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.conv_type = conv_type
        if dataformat == 'channels_last':
            self.channel_axis = -1
        else:
            raise NotImplementedError(
                'Only channels_last dataformat implemented. Got %s' % dataformat)
        
    def build(self, input_shape):
        if type(self.block_args) is not BlockArgs:
            self.block_args = BlockArgs(**self.block_args)
        if self.activation == 'swish':
            self.activation = tf.nn.swish
            self.l_activation = layers.Activation(self.activation)
        else:
            self.l_activation = layers.Activation(self.activation)
        
        if self.conv_type == 'conv':
            self.conv_cls = layers.Conv2D
        elif self.conv_type == 'separableconv':
            self.conv_cls = layers.SeparableConv2D
        else:
            raise ValueError('conv_type %s not supported' % self.conv_type)
        
        self.conv0 = self.conv_cls(self.block_args.output_filters,
                                   (self.block_args.kernel_size, 1),
                                   strides=(self.block_args.strides, 1),
                                   dilation_rate=(self.block_args.dilation, 1),
                                   padding=self.padding,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   use_bias=False
        )
        self.bn0 = layers.BatchNormalization()
        self.conv1 = self.conv_cls(self.block_args.output_filters,
                                   (self.block_args.kernel_size, 1),
                                   strides=(self.block_args.strides, 1),
                                   dilation_rate=(self.block_args.dilation, 1),
                                   padding=self.padding,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   use_bias=False
        )
        self.bn1 = layers.BatchNormalization(name='bn1')
        
        if self.dropout is not None:
            self.l_dropout = layers.Dropout(self.dropout)
        
        if self.block_args.pool_size > 1:
            self.maxpool = layers.MaxPooling2D(pool_size=(self.block_args.pool_size, 1))
        
        if self.block_args.se_ratio is not None:
            # Define the squeeze&excitation layers
            num_reduced_filters = max(
                1, int(self.block_args.output_filters*self.block_args.se_ratio))
            self.se_avg_pool = layers.GlobalAvgPool2D()
            self.se_reduce = layers.Conv1D(
                num_reduced_filters, 1,
                activation=self.activation,
                strides=1,
                padding='same',
                kernel_initializer=CONV_KERNEL_INITIALIZER
            )
            self.se_expand = layers.Conv1D(
                self.block_args.output_filters, 1,
                activation='sigmoid',
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding='same'
            )
        
    def compute_output_shape(self, input_shape):
        input_length = list(input_shape)[1]
        output_length = math.ceil(input_length / self.block_args.pool_size)
        output_shape = tensor_shape.TensorShape([
            input_shape[0],
            output_length,
            self.block_args.output_filters
        ])
        return output_shape
    
    def get_config(self):
        conf = super().get_config()
        conf.update({
            'block_args': self.block_args._asdict(),
            'activation': self.activation,
            'channel_axis': self.channel_axis,
            'padding': self.padding,
            'dropout': self.dropout,
            'conv_type': self.conv_type
        })
        return conf
    
    def _call_se(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()
        se_shape = (1, input_shape[self.channel_axis])
        x = self.se_avg_pool(input_tensor)
        x = layers.Reshape(se_shape)(x)
        x = self.se_reduce(x)
        x = self.se_expand(x)
        return layers.multiply([x, input_tensor])
    
    def call(self, inputs, training=None):
        # Debugging: raise error if image data format is channels_first
        if tf.keras.backend.image_data_format() != 'channels_last':
            raise ValueError('Keras data format is not channels last, but is %s' %
                             tf.keras.backend.image_data_format())
        # Broadcast inputs to 4D to use conv2d
        inputs = tf.expand_dims(inputs, axis=2)

        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.l_activation(x)
        
        x = self.conv1(x)
        bn1 = self.l_activation(self.bn1(x, training=training))
        if self.dropout is not None:
            bn1 = self.l_dropout(bn1, training=training)
        output = bn1
        
        # Squeeze and excite
        if self.block_args.se_ratio is not None:
            output = self._call_se(output)
            
        if self.block_args.pool_size > 1:
            output = self.maxpool(output)

        # Squeeze the expanded dim
        output = tf.squeeze(output, axis=2)
        return output, bn1
    
    
class Encoder(layers.Layer):
    """Encoder for u-time."""
    def __init__(self,
                 block_args_list=DEFAULT_BLOCK_ARGS,
                 activation='relu',
                 dropout=None,
                 conv_type='conv',
                 **kwargs):
        super().__init__(**kwargs)
        if type(block_args_list[0]) is not BlockArgs:
            # Transform dict arguments to BlockArgs namedtuples
            block_args_list = [BlockArgs(**arg_dict) for arg_dict in block_args_list]
        self.block_args_list = block_args_list
        self.activation = activation
        self.dropout = dropout
        self.conv_type = conv_type
        self.conv_blocks = []
        
    def build(self, input_shape):
        if type(self.block_args_list[0]) is not BlockArgs:
            # Transform dict arguments to BlockArgs namedtuples
            self.block_arg_lists = [BlockArgs(**arg_dict) for arg_dict in self.block_args_list]
        for block_args in self.block_args_list:
            cb = Conv1DBlock(
                block_args,
                activation=self.activation,
                dropout=self.dropout,
                conv_type=self.conv_type
            )
            self.conv_blocks.append(cb)
    
    def call(self, inputs, training=None):
        x = inputs
        residuals = []
        for cb in self.conv_blocks:
            x, residual = cb(x, training=training)
            residuals.append(residual)
        
        return [x, residuals[:-1]]
    
    def get_config(self):
        conf = super().get_config()
        conf.update({
            'block_args_list': [block_args._asdict() for block_args in self.block_args_list],
            'activation': self.activation,
            'dropout': self.dropout,
            'conv_type': self.conv_type
        })
        return conf
    
    def compute_output_shape(self, input_shape):
        input_length = list(input_shape)[1]
        output_length = math.ceil(input_length / self.block_args_list[0].pool_size)
        for block_args in self.block_args_list[1:]:
            output_length = math.ceil(output_length / block_args.pool_size)
        output_shape = tensor_shape.TensorShape([
            input_shape[0],
            output_length,
            self.block_args_list[-1].output_filters
        ])
        return output_shape
    
    
class Upsampling1DBlock(layers.Layer):
    
    def __init__(self,
                 block_args,
                 activation='relu',
                 padding='same',
                 data_format='channels_last',
                 dropout=None,
                 residual_learning=False,
                 conv_type='conv',
                 **kwargs):
        super().__init__(**kwargs)
        if type(block_args) is not BlockArgs:
            block_args = BlockArgs(**block_args)
        self.block_args = block_args
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.residual_learning = residual_learning
        self.conv_type = conv_type
        
        if data_format == 'channels_last':
            self.channel_axis = -1
        else:
            raise NotImplementedError(
                'Only channels_last dataformat implemented. Got %s' % data_format)
        
    def build(self, input_shape):
        if type(self.block_args) is not BlockArgs:
            self.block_args = BlockArgs(**self.block_args)
            
        if self.activation == 'swish':
            self.activation = tf.nn.swish
            self.l_activation = layers.Activation(self.activation)
        else:
            self.l_activation = layers.Activation(self.activation)
            
        if self.conv_type == 'conv':
            self.conv_cls = layers.Conv2D
        elif self.conv_type == 'separableconv':
            self.conv_cls = layers.SeparableConv2D
        else:
            raise ValueError('conv_type %s not supported' % self.conv_type)

        self.up = layers.UpSampling2D(size=(self.block_args.pool_size, 1), interpolation='bilinear')
        self.conv0 = self.conv_cls(self.block_args.output_filters,
                                   # TODO: Why is the pool size used here in the original u-time?
                                   (self.block_args.pool_size, 1),
                                   padding=self.padding,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.bn0 = layers.BatchNormalization()
        
        self.pad_nodes_to_match = layers.Lambda(lambda ns: pad_nodes_to_match(ns[0], ns[1]),
                                                name='pad_nodes_to_match')
        
        self.merge = layers.Concatenate(axis=-1)
        self.conv1 = self.conv_cls(self.block_args.output_filters,
                                   (self.block_args.kernel_size, 1),
                                   padding=self.padding,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = self.conv_cls(self.block_args.output_filters,
                                   (self.block_args.kernel_size, 1),
                                   padding=self.padding,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.bn2 = layers.BatchNormalization()
        
        if self.dropout is not None:
            self.l_dropout = layers.Dropout(self.dropout)
        
        if self.block_args.se_ratio is not None:
            # Define the squeeze&excitation layers
            num_reduced_filters = max(1, int(self.block_args.output_filters*self.block_args.se_ratio))
            self.se_avg_pool = layers.GlobalAvgPool2D()
            self.se_reduce = layers.Conv1D(
                num_reduced_filters, 1,
                activation=self.activation,
                strides=1,
                padding='same',
                kernel_initializer=CONV_KERNEL_INITIALIZER
            )
            self.se_expand = layers.Conv1D(
                self.block_args.output_filters, 1,
                activation='sigmoid',
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding='same'
            )
        
    def get_config(self):
        conf = super().get_config()
        conf.update({
            'block_args': self.block_args._asdict(),
            'activation': self.activation,
            'padding': self.padding,
            'dropout': self.dropout,
            'residual_learning': self.residual_learning,
            'conv_type': self.conv_type
        })
        return conf
    
    def compute_output_shape(self, input_shape):
        input_length = list(input_shape)[1]
        output_length = math.floor(input_length * self.block_args.pool_size)
        output_shape = tensor_shape.TensorShape([
            input_shape[0],
            output_length,
            self.block_args.output_filters
        ])
        return output_shape
    
    def _call_se(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()
        se_shape = (1, input_shape[self.channel_axis])
        x = self.se_avg_pool(input_tensor)
        x = layers.Reshape(se_shape)(x)
        x = self.se_reduce(x)
        x = self.se_expand(x)
        return layers.multiply([x, input_tensor])
        
    def call(self, inputs, training=None):
        inputs, residuals = inputs
        inputs = tf.expand_dims(inputs, axis=2)
        
        x = self.up(inputs)
        x = self.conv0(x)
        x = self.bn0(x, training=training)
        x = self.l_activation(x)
        
        # Do padding for residuals if pooling has implicitly cropped the outputs
        # along the encoder path
        x = self.pad_nodes_to_match([residuals, x])
        
        x = self.merge([x, residuals])
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.l_activation(x)
        x = self.conv2(x)
        output = self.l_activation(self.bn2(x, training=training))
        if self.dropout is not None:
            output = self.l_dropout(output, training=training)
        
        # Squeeze and excite
        if self.block_args.se_ratio is not None:
            output = self._call_se(output)
        
        # Squeeze the expanded dim (because conv2d was used)
        output = tf.squeeze(output, axis=2)
        
        return output
    
    
class Decoder(layers.Layer):
    def __init__(self,
                 block_args_list=DEFAULT_BLOCK_ARGS,
                 activation='relu',
                 dropout=None,
                 conv_type='conv',
                 **kwargs):
        super().__init__(**kwargs)
        if type(block_args_list[0]) is not BlockArgs:
            # Transform dict arguments to BlockArgs namedtuples
            block_args_list = [BlockArgs(**arg_dict) for arg_dict in block_args_list]
        self.block_args_list = block_args_list
        self.activation = activation
        self.dropout = dropout
        self.conv_type = conv_type
        self.up_blocks = []
        
    def build(self, input_shape):
        if type(self.block_args_list[0]) is not BlockArgs:
            # Transform dict arguments to BlockArgs namedtuples
            self.block_args_list = [BlockArgs(**arg_dict) for arg_dict in self.block_args_list]
        # Don't do squeeze and excitation on final output, since it is done separately for each
        # segmentation output.
        self.block_args_list[0] = self.block_args_list[0]._replace(se_ratio=None)
        
        # Exclude the bottom block, iterate reverse
        for block_args in self.block_args_list[:-1][::-1]:
            ub = Upsampling1DBlock(
                block_args,
                activation=self.activation,
                dropout=self.dropout,
                conv_type=self.conv_type
            )
            self.up_blocks.append(ub)
    
    def get_config(self):
        conf = super().get_config()
        conf.update({
            'block_args_list': [block_args._asdict() for block_args in self.block_args_list],
            'activation': self.activation,
            'dropout': self.dropout,
            'conv_type': self.conv_type
        })
        return conf
    
    def compute_output_shape(self, input_shape):
        input_length = list(input_shape)[1]
        output_length = math.floor(input_length * self.block_args_list[-1].pool_size)
        for block_args in self.block_args_list[1::-1]:
            output_length = math.floor(output_length * block_args.pool_size)
        output_shape = tensor_shape.TensorShape([
            input_shape[0],
            output_length,
            self.block_args_list[0].filters
        ])
        return output_shape
    
    def call(self, inputs, training=None):
        x, residuals = inputs
        for ub, residual in zip(self.up_blocks, residuals[::-1]):
            x = ub([x, residual], training=training)
            
        return x
    
    
def UTimeF(input_shape,
           block_args_list=DEFAULT_BLOCK_ARGS,
           name='u-time',
           activation='relu',
           enc_cls=Encoder,
           dec_cls=Decoder,
           conv_type='conv',
           drop_rate=None,
           aspp_depth=None,
           aspp_dropout=None,
           aspp_conv_cls=layers.Conv1D,
           segment_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
           input_names=None,
           output_args=DEFAULT_OUTPUT_ARGS,
           clf_se_ratio=0.25):
    """A function to setup a u-time based model.
    
    Args:
        input_shape: The shape of individual input signals.
            (None, 1) for arbitrary length.
        block_args_list: A list of BlockArgs items configuring each block
            of the encoder. Iterated in reverse order to create the decoder.
        name: The name of the Keras model to be returned.
        activation: The activation function used throughout the model.
        enc_cls: The class used to instantiate the encoder.
        dec_cs: The class used to instantiate the decoder.
        conv_type: The type of convolution used. 'conv' for Conv,
            'separableconv' for SeparableConv.
        drop_rate: The dropout rate used in the encoder and decoder.
        aspp_depth: The number of features in each branch of the ASPP block.
        aspp_dropout: The dropout rate used in the ASPP block.
        aspp_conv_cls: The class used to instantiate convs in the ASPP block.
        segment_kernel_regularizer: The regularizer used in the final layer.
        input_names: The names used for the input signals. If a list, there are
            multiple input signals. If not, considered as the name of a single input.
        output_args: A lits of OutputArgs used to configure each output,
            including the output segment classifier.
        clf_se_ratio: The squeeze&excitation ratio used before the segment classifier.
    
    Returns:
        A keras Model.
    """
    input_shape = list(input_shape)
    
    if activation == 'relu':
        activation = tf.nn.relu
    
    if type(block_args_list[0]) is not BlockArgs:
        # Transform dict arguments to BlockArgs namedtuples
        block_args_list = [BlockArgs(**arg_dict) for arg_dict in block_args_list]
        
    if type(output_args[0]) is not OutputArgs:
        # Transform dict arguments to OutputArgs namedtuples
        output_args = [OutputArgs(**arg_dict) for arg_dict in output_args]
    
    #### Define and concatenate inputs ####
    if type(input_names) is list:
        inputs = []
        nchannels = len(input_names)
        for input_name in input_names:
            inp = layers.Input(shape=input_shape, name=input_name)
            inputs.append(inp)
        if nchannels > 1:
            enc_inputs = layers.Concatenate(name='concatenated')(inputs)
        else:
            enc_inputs = inputs[0]
    else:
        inputs = layers.Input(shape=input_shape, name=input_names)
        enc_inputs = inputs
    
    #### Encoder ####
    encoded, residuals = enc_cls(
        block_args_list, activation=activation,
        dropout=drop_rate, conv_type=conv_type, name='encoder')(enc_inputs)
    
    #### ASPP ####
    if aspp_depth is not None:
        encoded = ASPP(encoded, depth=aspp_depth, activation=activation, conv_cls=aspp_conv_cls)
        if aspp_dropout is not None:
            encoded = layers.Dropout(aspp_dropout)(encoded)
    
    #### Decoder ####
    decoded = dec_cls(
        block_args_list, activation=activation,
        dropout=drop_rate, conv_type=conv_type, name='decoder')([encoded, residuals])

    #### Define the outputs ####
    outputs = []
    for ocfg in output_args:
        if clf_se_ratio is not None:
            _decoded = squeeze_and_excite_1d(decoded, se_ratio=clf_se_ratio, activation=activation)
        else:
            _decoded = decoded
        out = layers.Conv2D(filters=ocfg.n_classes,
                            kernel_size=(ocfg.dense_ksize, 1),
                            activation=ocfg.dense_activation,
                            padding='same',
                            name='%s_%s' % (ocfg.output_name, 'dense_pred')
                            )(tf.expand_dims(_decoded, axis=2))
        if ocfg.samples_per_segment is not None:
            out = layers.AveragePooling2D((ocfg.samples_per_segment, 1),
                                          name='%s_%s'% (ocfg.output_name, 'segment_pool'))(out)
            out = layers.Conv2D(filters=ocfg.n_classes,
                                kernel_size=(ocfg.segment_ksize, 1),
                                activation=ocfg.segment_activation,
                                kernel_regularizer=segment_kernel_regularizer,
                                padding='same',
                                name='%s_%s' % (ocfg.output_name, 'segment_pred'))(out)

        out = layers.Reshape([-1, out.shape[-1]], name=ocfg.output_name)(out)
        outputs.append(out)

    return Model(inputs=inputs, outputs=outputs, name=name)
