import tensorflow as tf

class AE:
    
    def __init__(self, 
        encoder_model, decoder_model, 
        name = 'model', verbose=True
    ):
        '''
        Default constructor for Convolutional Autoencoder
        '''
        self.verbose  = verbose
        self.model_name = name
        self.model, self.encoder, self.decoder = self.get_model(encoder_model, decoder_model)
        
        self.loss_tracks = None
        self.loss_tracks_total = None
        

    def get_model(self, encoder_model, decoder_model):
        """Returns the encoder, decoder, and models"""
        encoder = encoder_model(name=f'enc_{self.model_name}')
        if self.verbose: 
            print("Encoder Structure:") 
            encoder.summary()
        decoder = decoder_model(name=f'dec_{self.model_name}')
        if self.verbose: 
            print("Decoder Structure:")
            decoder.summary()
        inps = encoder.inputs
        outs = decoder(encoder(inps))
        model = tf.keras.Model(inps, outs, name=f'ae_{self.model_name}')
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
            stat_tracker.save_cp([self.model, self.encoder, self.decoder], ['all', 'enc', 'dec'], optimizer)

            # Iterate over the batches of the dataset.
            for step, inp in enumerate(train_ds[0]):

                if step % max(1, len(train_ds)//8) == 0: print('.', end='')
                loss = 0

                stat_tracker.set_epoch_step(epoch, step)
                stat_tracker.log(f"Getting predictions from model")
                
                with tf.GradientTape() as tape:
                    z = self.encoder(inp)
                    out = self.decoder(z)
                    
                    for name, lf in losses.items():
                        c_loss = lf(inp, out, [z])
                        self.loss_tracks[name](c_loss)
                        loss += c_loss
                    
                    self.loss_tracks_total(loss)
                
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
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


class TAE(AE):

    """
    First iteration of the 'targetted' autoencoder from notebook
    Allows operation encoding that is fed in line with latent bottleneck features
    """

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
            stat_tracker.save_cp([self.model, self.encoder, self.decoder], ['all', 'enc', 'dec'], optimizer)

            # Iterate over the batches of the dataset.
            for step, (inp, lab, out) in enumerate(zip(*train_ds)):

                if step % max(1, len(train_ds)//8) == 0: print('.', end='')
                loss = 0

                stat_tracker.set_epoch_step(epoch, step)
                stat_tracker.log(f"Getting predictions from model")
                
                with tf.GradientTape() as tape:
                    z = self.encoder([inp, lab])
                    pred = self.decoder(z)
                    
                    for name, lf in losses.items():
                        c_loss = lf(out, pred, [z])
                        self.loss_tracks[name](c_loss)
                        loss += c_loss
                    
                    self.loss_tracks_total(loss)
                
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
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


class TAE2():

    """
    Second iteration of 'Targetted Autoencoder'. 
    Includes additional loss term that forces decoder to switch to identity function with only 
    one dense layer to figure it out. This ensures that the bottleneck features reach the output 
    layer so that the decoder (hopefully) doesn't just memorize the one-hot-encoding correlation.
    """
    
    def __init__(self, 
        encoder_model, decoder_model, 
        name = 'model', verbose=True
    ):
        '''
        Default constructor for Convolutional Autoencoder
        '''
        self.verbose  = verbose
        self.model_name = name
        self.model, self.model_eval, self.encoder, self.decoder = self.get_model(encoder_model, decoder_model)
        
        self.loss_tracks = None
        self.loss_tracks_total = None
        

    def get_model(self, encoder_model, decoder_model):
        """Returns the encoder, decoder, and models"""
        encoder = encoder_model(name=f'enc_{self.model_name}')
        if self.verbose: 
            print("Encoder Structure:") 
            encoder.summary()
        decoder = decoder_model(name=f'dec_{self.model_name}')
        if self.verbose: 
            print("Decoder Structure:")
            decoder.summary()
        inps = encoder.inputs
        outs = decoder(encoder(inps))
        models = [
            tf.keras.Model(inps, outs[0], name=f'tae_{self.model_name}'),
            tf.keras.Model(inps, outs, name=f'tae_{self.model_name}')
        ]
        return *models, encoder, decoder

    def train(self, 
        train_set, epochs, optimizer, losses, test_samples, stat_tracker, load_cp = True
    ):
        """Training loop. Display generated images each epoch"""
        start_epoch = stat_tracker.load_cp([self.model, self.encoder, self.decoder], optimizer) if load_cp else 0
        
        self.loss_tracks =      {f'{k}_trans' : tf.keras.metrics.Mean(name=f'{k}_trans') for k in losses.keys()}
        self.loss_tracks.update({f'{k}_mimic' : tf.keras.metrics.Mean(name=f'{k}_mimic') for k in losses.keys()})
        self.loss_tracks_total = tf.keras.metrics.Mean(name='loss_total')

        stat_tracker.log(f"Begin Training Model '{self.model_name}'")

        for epoch in range(start_epoch, epochs + 1):
            print(f'Start of epoch {epoch}', end='')
            train_ds = train_set() if hasattr(train_set, '__call__') else train_set

            stat_tracker.set_epoch_step(epoch, 0)
            stat_tracker.save_cp([self.model, self.encoder, self.decoder], ['all', 'enc', 'dec'], optimizer)

            # Iterate over the batches of the dataset.
            for step, (inp, lab, out) in enumerate(zip(*train_ds)):

                if step % max(1, len(train_ds)//8) == 0: print('.', end='')
                loss = 0

                stat_tracker.set_epoch_step(epoch, step)
                stat_tracker.log(f"Getting predictions from model")
                
                with tf.GradientTape() as tape:
                    z = self.encoder([inp, lab])
                    preds = self.decoder(z)
                    
                    for name, lf in losses.items():
                        c_loss = lf(out, preds[0], [z])
                        self.loss_tracks[f'{name}_trans'](c_loss)
                        loss += c_loss
                    
                    for name, lf in losses.items():
                        c_loss = lf(inp, preds[1], [z])
                        self.loss_tracks[f'{name}_mimic'](c_loss)
                        loss += c_loss
                        
                    self.loss_tracks_total(loss)
                
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

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