import tensorflow as tf
from functools import partial

class LGAN:
    
    def __init__(self, 
        gen_model, dis_model, 
        name = 'model', verbose=True
    ):
        '''Default constructor for Latent GAN'''
        self.verbose  = verbose
        self.model_name = name
        self.model_g, self.model_d = self.get_model(gen_model, dis_model)
        
        self.loss_tracks = None
        self.loss_tracks_total = None


    def get_model(self, gen_model, dis_model):
        """Returns the generator and discriminator models"""
        model_g = gen_model(f'gen_model_{self.model_name}')
        if self.verbose: 
            print("Generator Structure:") 
            model_g.summary()
        model_d = dis_model(f'dis_model_{self.model_name}')
        if self.verbose: 
            print("Discriminator Structure:")
            model_d.summary()
        return model_g, model_d


    def gradient_penalty(self, f, real, fake, mode):
        """
        Gradient penalty function to conform to WGAN/DRAGAN standards
        From course content lab. Associated colab link:  
        https://colab.research.google.com/drive/1zAUGSNFENZ_iU7m8YkiniG8seUNqbYT5#scrollTo=gKNwOru-2_Vb
        """
        def _gradient_penalty(f, real, fake=None):
            def _interpolate(a, b=None):
                if b is None:   # interpolation in DRAGAN
                    beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                    b = a + 0.5 * tf.math.reduce_std(a) * beta
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.shape)
                return inter

            x = _interpolate(real, fake)
            with tf.GradientTape() as t:
                t.watch(x)
                pred = f(x)
            grad = t.gradient(pred, x)
            norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
            gp = tf.reduce_mean((norm - 1.)**2)

            return gp

        if mode == 'dragan':
            return _gradient_penalty(f, real)
        elif mode == 'wgan-gp':
            return _gradient_penalty(f, real, fake)
        return tf.constant(0, dtype=real.dtype)


    def train(self, 
        train_set, epochs, optimizers, losses, test_samples, stat_tracker, gp_mode='none', gp_weight=0.5, train_counts=(1,1), load_cp=True
    ):
        """Training loop. Display generated images each epoch"""
        start_epoch = stat_tracker.load_cp([self.model_g, self.model_d], optimizers) if load_cp else 0

        loss_names = list(losses[0].keys()) + list(losses[1].keys())
        if gp_mode != 'none': loss_names += [gp_mode.upper()]
        
        self.loss_tracks = {k : tf.keras.metrics.Mean(name=k) for k in loss_names}
        self.loss_tracks_total = tf.keras.metrics.Mean(name='loss_total')

        stat_tracker.log(f"Begin Training Model '{self.model_name}'")

        for epoch in range(start_epoch, epochs+1):
            print(f'Start of epoch {epoch}', end='')
            train_ds = train_set() if hasattr(train_set, '__call__') else train_set

            stat_tracker.set_epoch_step(epoch, 0)
            stat_tracker.save_cp([self.model_g, self.model_d], ['gen', 'dis'], optimizers)

            # Iterate over the batches of the dataset.
            for step, inp in enumerate(train_ds[0]):

                if step % max(1, len(train_ds)//8) == 0: print('.', end='')
                stat_tracker.set_epoch_step(epoch, step)
                d_loss = g_loss = 0
                
                stat_tracker.log(f"Training Discriminator")
                for _ in range(train_counts[0]):
                    with tf.GradientTape() as d_tape:
                        self.model_g.trainable = False
                        self.model_d.trainable = True

                        noise = tf.random.normal(shape=(len(inp), *self.model_g.inputs[0][0].shape))
                        true_inp = inp
                        fake_inp = self.model_g(noise)
                        true_out = self.model_d(true_inp)
                        fake_out = self.model_d(fake_inp)
                        
                        for name, lf in losses[0].items():
                            c_loss = lf(fake_out, true_out, self.model_d) / 2
                            self.loss_tracks[name](c_loss)
                            d_loss += c_loss

                        gp = self.gradient_penalty(
                            partial(self.model_d, training=True), 
                            true_inp, fake_inp, 
                            mode=gp_mode
                        )
                        gp *= gp_weight / (len(inp) * 2)
                        if gp_mode != 'none': 
                            self.loss_tracks[gp_mode.upper()](gp)
                        d_loss += gp

                    grads = d_tape.gradient(d_loss, self.model_d.trainable_weights)
                    optimizers[0].apply_gradients(zip(grads, self.model_d.trainable_weights))

                stat_tracker.log(f"Training Generator")
                for _ in range(train_counts[1]):
                    with tf.GradientTape() as g_tape:
                        self.model_g.trainable = True
                        self.model_d.trainable = False

                        noise = tf.random.normal(shape=(len(inp), *self.model_g.inputs[0][0].shape))
                        fake_out = self.model_d(self.model_g(noise))
                        
                        for name, lf in losses[1].items():
                            c_loss = lf(fake_out, tf.ones_like(fake_out), self.model_g) / 2
                            self.loss_tracks[name](c_loss)
                            g_loss += c_loss

                    grads = g_tape.gradient(g_loss, self.model_g.trainable_weights)
                    optimizers[1].apply_gradients(zip(grads, self.model_g.trainable_weights))

                self.loss_tracks_total(g_loss + d_loss)

            mean_loss = self.loss_tracks_total.result().numpy()
            avg_losses = {k : v.result().numpy() for k,v in self.loss_tracks.items()}

            print(f'\nEpoch: {epoch} | mean loss = {mean_loss:.8f}', end='')
            losses_str = ' | '.join([f'{k} : {v:.5}' for k,v in avg_losses.items()])
            print(f'\t( {losses_str} )')
            stat_tracker.log(f"mean loss = {self.loss_tracks_total.result().numpy():.8f}\t( {losses_str} )")
            stat_tracker.write_loss(list(avg_losses.items()) + [('ALL', mean_loss)])

            stat_tracker.log(f"Starting Visualizations")
            stat_tracker.vis_save([self.model_g, self.model_d], test_samples)
            stat_tracker.log(f" - Finished\n")
            
            self.loss_tracks_total.reset_state()
            [t.reset_state() for t in self.loss_tracks.values()]