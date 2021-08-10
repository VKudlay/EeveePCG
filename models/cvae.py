import tensorflow as tf

class VAE:
    
    def __init__(self, 
        encoder_model, decoder_model, 
        name = 'model', verbose = True
    ):
        '''
        Default constructor for Convolutional Autoencoder
        :param in_shape: input shape of features
        :param n_botl: number of latent features in bottleneck
        :param num_conv: number of convolutions per layer (assuming enc/dec symmetry)
        '''
        self.verbose = verbose
        self.model_name = name
        self.model, self.encoder, self.decoder = self.get_model(encoder_model, decoder_model)
        
        self.loss_tracks = None
        self.loss_tracks_total = None

    def get_model(self, encoder_model, decoder_model):
        """Returns the encoder, decoder, and models"""
        encoder = encoder_model(f'enc_model_{self.model_name}')
        if self.verbose: 
            print("Encoder Structure:") 
            encoder.summary()
        decoder = decoder_model(f'dec_model_{self.model_name}')
        if self.verbose: 
            print("Decoder Structure:")
            decoder.summary()
        inps = encoder.inputs
        outs = decoder(encoder(inps)[0])
        model = tf.keras.Model(inps, outs, name=f'vae_{self.model_name}')
        return model, encoder, decoder


    def train(self, 
        train_set, epochs, optimizer, losses, test_samples, stat_tracker, load_cp = True
    ):
        """Training loop. Display generated images each epoch"""
        start_epoch = stat_tracker.load_cp([self.model, self.encoder, self.decoder], optimizer) if load_cp else 0
        
        self.loss_tracks = {k : tf.keras.metrics.Mean(name=k) for k in losses.keys()}
        self.loss_tracks_total = tf.keras.metrics.Mean(name='loss_total')

        stat_tracker.log(f"Begin Training Model '{self.model_name}'")

        for epoch in range(start_epoch, epochs + 1):
            print(f'Start of epoch {epoch}', end='')
            train_ds = train_set() if hasattr(train_set, '__call__') else train_set

            stat_tracker.set_epoch_step(epoch, 0)
            stat_tracker.log(f"Saving Model Checkpoint")
            stat_tracker.save_cp([self.model, self.encoder, self.decoder], ['all', 'enc', 'dec'], optimizer)
            stat_tracker.log(f" - Finished")

            # Iterate over the batches of the dataset.
            for step, inp in enumerate(train_ds[0]):

                if step % max(1, len(train_ds)//8) == 0: print('.', end='')
                loss = 0

                stat_tracker.set_epoch_step(epoch, step)
                stat_tracker.log(f"Opening tape")
                
                with tf.GradientTape() as tape:
                    stat_tracker.log(f"Getting predictions from model")
                    mu, sd, z = self.encoder(inp)
                    out = self.decoder(z)

                    for name, lf in losses.items():
                        c_loss = lf(inp, out, (z, mu, sd))
                        self.loss_tracks[name](c_loss)
                        loss += c_loss
                    
                    self.loss_tracks_total(loss)
                
                stat_tracker.log(f"Calculating Grads")
                grads = tape.gradient(loss, self.model.trainable_weights)
                stat_tracker.log(f"Applying Grads")
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                stat_tracker.log(f" - Finished\n")
                self.loss_tracks_total(loss)
            
            mean_loss = self.loss_tracks_total.result().numpy()
            avg_losses = {k : v.result().numpy() for k,v in self.loss_tracks.items()}

            print(f'\nEpoch: {epoch} | mean loss = {mean_loss:.8f}', end='')
            losses_str = ' | '.join([f'{k} : {v:.5}' for k,v in avg_losses.items()])
            print(f'\t( {losses_str} )')
            stat_tracker.log(f"mean loss = {self.loss_tracks_total.result().numpy():.8f}\t( {losses_str} )")
            stat_tracker.write_loss(list(avg_losses.items()) + [('ALL', mean_loss)])

            stat_tracker.log(f"Starting Visualizations")
            stat_tracker.vis_save([self.model, self.encoder, self.decoder], test_samples)
            stat_tracker.log(f" - Finished\n")
            
            self.loss_tracks_total.reset_state()
            [t.reset_state() for t in self.loss_tracks.values()]