'''
File specifying the models that will be used throughout the experiments.
'''

import tensorflow as tf

class AE_DefaultModel:

    def __init__(self, n_botl, n_samp):
        """AE_DefaultModel Default Constructor
        :param n_botl: number of latent features in the bottleneck
        :param n_samp: number of point samples in the point clouds
        """
        self.n_botl = n_botl
        self.n_samp = n_samp

    def encoder_model(self, name='enc_model'):
        """Returns the encoder model with `encoder_layers` layers"""
        inps = tf.keras.layers.Input(shape=(self.n_samp, 3))
        return tf.keras.Model(inps, self.encoder_layers(inps), name=name)
    
    def decoder_model(self, name='dec_model'):
        """Returns the decoder model with `decoder_layers` layers"""
        inps = tf.keras.layers.Input(shape=(self.n_botl,))
        return tf.keras.Model(inps, self.decoder_layers(inps), name=name)

    def encoder_layers(self, inps):
        """Default layer specification for default autoencoder encoder
        
        Description: 1-stride 1-kernel size convolutional layers with {64, 128, 128, 256, 128} 
        filters, each followed by a max pooling and batch normalization layer.
        Finally, a feature-wise max layer for final latent bottleneck representation. 
        """
        def ConvSet(f, name, k=1, s=1, a='relu'):
            def tmp_fn(x):
                xt = tf.keras.layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=a, name=f'enc_conv_{name}')(x)
                xt = tf.keras.layers.MaxPooling1D(name=f'enc_maxp_{name}')(xt)
                return tf.keras.layers.BatchNormalization(name=f'enc_bn_{name}')(xt)
            return tmp_fn
        
        x = ConvSet( 64, name=1)(inps)
        x = ConvSet(128, name=2)(x)
        x = ConvSet(128, name=3)(x)
        x = ConvSet(256, name=4)(x)
        x = ConvSet(128, name=5)(x)
        x = tf.math.reduce_max(x, axis=1)
        
        return x

    def decoder_layers(self, inps):
        """Default layer specification for default autoencoder decoder"""
        # feed to a Dense network with units computed from the conv_shape dimensions
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense1")(inps)        
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense2")(x)        
        x = tf.keras.layers.Dense(self.n_samp * 3, activation = 'sigmoid', name="decode_final_dense")(x)
        x = tf.keras.layers.Reshape((self.n_samp, 3), name="decode_final_reshape")(x)

        return x


class LGAN_DefaultModel:

    def __init__(self, n_botl, n_samp, n_input):
        """LGAN_DefaultModel Default Constructor
        :param n_botl: number of latent features in the bottleneck
        :param n_samp: number of point samples in the point clouds
        """
        self.n_botl  = n_botl
        self.n_samp  = n_samp
        self.n_input = n_input

    def gen_model(self, name='gen_model'):
        """Defines the generator model with gen_layers layers"""
        inps = tf.keras.layers.Input(shape=self.n_input)
        layers = self.gen_layers(inps)
        model = tf.keras.Model(inps, layers, name=name)
        return model

    def dis_model(self, name='dis_model'):
        """Defines the critic model with dis_layers layers"""
        inps = tf.keras.layers.Input(shape=self.n_botl)
        layers = self.dis_layers(inps)
        model = tf.keras.Model(inps, layers, name=name)
        return model

    def gen_layers(self, inps):
        x = tf.keras.layers.Dense(128, activation='relu', name="gen_dense_0")(inps)
        x = tf.keras.layers.Dense(128, activation='relu', name="gen_dense_1")(x)
        return x

    def dis_layers(self, inps):
        x = tf.keras.layers.Dense(256, activation='relu', name="dis_dense_0")(inps)
        x = tf.keras.layers.Dense(512, activation='relu', name="dis_dense_1")(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name="dis_out")(x)
        return x


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Generates a random sample and combines with the encoder output

        Args:
          inputs -- output tensor from the encoder

        Returns:
          `inputs` tensors combined with a random sample
        """
        mu, var = inputs
        batch  = tf.shape(mu)[0]
        dim    = tf.shape(mu)[1]
        eps    = tf.keras.backend.random_normal(shape=(batch, dim))
        z      = mu + tf.exp(0.5 * var) * eps # effective in practice
        return  z

class VAE_DefaultModel:

    def __init__(self, n_botl, n_samp):
        self.n_botl = n_botl
        self.n_samp = n_samp

    def encoder_model(self, name='enc_model'):
        """Defines the encoder model with the Sampling layer"""
        inps = tf.keras.layers.Input(shape=(self.n_samp, 3))
        mu, sd = self.encoder_layers(inps)
        z = Sampling()((mu, sd))
        return tf.keras.Model(inps, [mu, sd, z], name=name)

    def decoder_model(self, name='dec_model'):
        """Returns the decoder model with `decoder_layers` layers"""
        inps = tf.keras.layers.Input(shape=(self.n_botl,))
        return tf.keras.Model(inps, self.decoder_layers(inps), name=name)

    def encoder_layers(self, inps):
        """Default layer specification for default autoencoder encoder
        
        Description: 1-stride 1-kernel size convolutional layers with {64, 128, 128, 256, 128} 
        filters, each followed by a max pooling and batch normalization layer.
        Finally, a feature-wise max layer for final latent bottleneck representation. 
        """
        def ConvSet(f, name, k=1, s=1, a='relu'):
            def tmp_fn(x):
                xt = tf.keras.layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=a, name=f'enc_conv_{name}')(x)
                xt = tf.keras.layers.MaxPooling1D(name=f'enc_maxp_{name}')(xt)
                return tf.keras.layers.BatchNormalization(name=f'enc_bn_{name}')(xt)
            return tmp_fn
        
        x = ConvSet( 64, name=1)(inps)
        x = ConvSet(128, name=2)(x)
        x = ConvSet(128, name=3)(x)
        x = ConvSet(256, name=4)(x)
        x = ConvSet(128, name=5)(x)
        x = tf.keras.layers.Flatten(name="encode_flatten")(x)
        mu = tf.keras.layers.Dense(self.n_botl, name='latent_mu')(x)
        sd = tf.keras.layers.Dense(self.n_botl, name='latent_sd')(x)
        
        return mu, sd

    def decoder_layers(self, inps):
        """Default layer specification for default autoencoder decoder"""
        # feed to a Dense network with units computed from the conv_shape dimensions
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense1")(inps)        
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense2")(x)        
        x = tf.keras.layers.Dense(self.n_samp * 3, activation = 'sigmoid', name="decode_final_dense")(x)
        x = tf.keras.layers.Reshape((self.n_samp, 3), name="decode_final_reshape")(x)

        return x



class TAE_DefaultModel:

    def __init__(self, n_botl, n_samp, n_cats):
        self.n_botl = n_botl
        self.n_samp = n_samp
        self.n_cats = n_cats

    def encoder_model(self, name='enc_model'):
        """Returns the encoder model with `encoder_layers` layers"""
        inps1 = tf.keras.layers.Input(shape=(self.n_samp, 3))
        inps2 = tf.keras.layers.Input(shape=(self.n_cats))
        inps = [inps1, inps2]
        return tf.keras.Model(inps, self.encoder_layers(inps), name=name)
    
    def decoder_model(self, name='dec_model'):
        """Returns the decoder model with `decoder_layers` layers"""
        inps = tf.keras.layers.Input(shape=(self.n_botl + self.n_cats))
        return tf.keras.Model(inps, self.decoder_layers(inps), name=name)

    def encoder_layers(self, inps):
        """Default layer specification for default autoencoder encoder
        
        Description: 1-stride 1-kernel size convolutional layers with {64, 128, 128, 256, 128} 
        filters, each followed by a max pooling and batch normalization layer.
        Finally, a feature-wise max layer for final latent bottleneck representation. 
        """
        def ConvSet(f, name, k=1, s=1, a='relu'):
            def tmp_fn(x):
                xt = tf.keras.layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=a, name=f'enc_conv_{name}')(x)
                xt = tf.keras.layers.MaxPooling1D(name=f'enc_maxp_{name}')(xt)
                return tf.keras.layers.BatchNormalization(name=f'enc_bn_{name}')(xt)
            return tmp_fn
        x, cat = inps
        x = ConvSet( 64, name=1)(x)
        x = ConvSet(128, name=2)(x)
        x = ConvSet(128, name=3)(x)
        x = ConvSet(256, name=4)(x)
        x = ConvSet(128, name=5)(x)
        x = tf.math.reduce_max(x, axis=1)
        return tf.concat([x, cat], axis=1)

    def decoder_layers(self, inps):
        """Default layer specification for default autoencoder decoder"""
        # feed to a Dense network with units computed from the conv_shape dimensions
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense1")(inps)        
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense2")(x)        
        x = tf.keras.layers.Dense(self.n_samp * 3, activation = 'sigmoid', name="decode_final_dense")(x)
        x = tf.keras.layers.Reshape((self.n_samp, 3), name="decode_final_reshape")(x)

        return x


class TAE2_DefaultModel:

    def __init__(self, n_botl, n_samp, n_cats):
        self.n_botl = n_botl
        self.n_samp = n_samp
        self.n_cats = n_cats

    def encoder_model(self, name='enc_model'):
        """Returns the encoder model with `encoder_layers` layers"""
        inps1 = tf.keras.layers.Input(shape=(self.n_samp, 3))
        inps2 = tf.keras.layers.Input(shape=(self.n_cats))
        inps = [inps1, inps2]
        return tf.keras.Model(inps, self.encoder_layers(inps), name=name)
    
    def decoder_model(self, name='dec_model'):
        """Returns the decoder model with `decoder_layers` layers"""
        inps = tf.keras.layers.Input(shape=(self.n_botl + self.n_cats))
        return tf.keras.Model(inps, self.decoder_layers(inps), name=f'{name}_dual')

    def encoder_layers(self, inps):
        """Default layer specification for default autoencoder encoder
        
        Description: 1-stride 1-kernel size convolutional layers with {64, 128, 128, 256, 128} 
        filters, each followed by a max pooling and batch normalization layer.
        Finally, a feature-wise max layer for final latent bottleneck representation. 
        """
        def ConvSet(f, name, k=1, s=1, a='relu'):
            def tmp_fn(x):
                xt = tf.keras.layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=a, name=f'enc_conv_{name}')(x)
                xt = tf.keras.layers.MaxPooling1D(name=f'enc_maxp_{name}')(xt)
                return tf.keras.layers.BatchNormalization(name=f'enc_bn_{name}')(xt)
            return tmp_fn
        x, cat = inps
        x = ConvSet( 64, name=1)(x)
        x = ConvSet(128, name=2)(x)
        x = ConvSet(128, name=3)(x)
        x = ConvSet(256, name=4)(x)
        x = ConvSet(128, name=5)(x)
        x = tf.math.reduce_max(x, axis=1)
        return tf.concat([x, cat], axis=1)

    def decoder_layers(self, inps):
        """Default layer specification for default autoencoder decoder"""
        # feed to a Dense network with units computed from the conv_shape dimensions
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense1")(inps)        
        x = tf.keras.layers.Dense(256 * 3, activation = 'relu', name="decode_dense2")(x)        
        trans = tf.keras.layers.Dense(self.n_samp * 3, activation = 'sigmoid', name="decode_trans_dense")(x)
        mimic = tf.keras.layers.Dense(self.n_samp * 3, activation = 'sigmoid', name="decode_mimic_dense")(x)
        trans = tf.keras.layers.Reshape((self.n_samp, 3), name="decode_trans_reshape")(trans)
        mimic = tf.keras.layers.Reshape((self.n_samp, 3), name="decode_mimic_reshape")(mimic)

        return (trans, mimic)