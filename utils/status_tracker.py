import matplotlib.pyplot as plt
import tensorflow as tf
import random
from utils.mesh_ops import plot_pointfield

import logging
import os
import time

import glob
from PIL import Image

def set_tf_logmin(level):
    '''Sets tensorflow alert level. Could be used as a general utility when 
        issue/message is understood but irrelevant/due to version-specific issue
    '''
    # Credits to https://stackoverflow.com/a/57439591/5003309
    level_num = ['info', 'warning', 'error', 'fatal'].index(level.lower())
    level_log = [logging.INFO, logging.WARNING, logging.ERROR, logging.FATAL][level_num]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level_num)
    logging.getLogger('tensorflow').setLevel(level_log)


def dir_to_gif(in_path, out_path, duration = 10, loop = 0):
    '''Convert a directory of images to a gif'''
    # Credits to https://stackoverflow.com/a/57751793/5003309
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(in_path))]
    img.save(fp=out_path, format='GIF', append_images=imgs,
            save_all=True, duration=duration, loop=loop)


class StatusTracker:
    '''Class for tracking status of run. Made because tf would occasionally crash. Thanks tf'''
    def __init__(self, model_name, vis_save_fn, vis_interval = 1, checkpoint_interval = 5, folder='model_runs'):
        '''Default constructor
        :param model_name: string specifying the directory to save the logger outputs
        :param vis_save_fn: function specifying how visuals are to be generated. See example
        :param vis_interval: how often should the visual be produced/displayed/saved
        :param checkpoint_interval: how often should the checkpoint be saved
        '''

        self.model_name = model_name
        self.vis_save_f = vis_save_fn
        self.vis_interv = vis_interval
        self.chp_interv = checkpoint_interval   
        
        self.output_dir = f'{folder}/{model_name}_output'
        self.picout_dir = f'{self.output_dir}/pics'
        self.state_file = f'{self.output_dir}/live_stat.txt'
        self.loss_file  = f'{self.output_dir}/loss_stat.txt'
        self.checkp_dir = f'{self.output_dir}/checkpoints'
        self.meta_file  = f'{self.output_dir}/meta.txt'

        self.checkpoints = []
        self.checkp_mgrs = []

        self.curr_epoch = None 
        self.curr_step  = None

        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        if not os.path.exists(self.picout_dir): os.makedirs(self.picout_dir)
        if not os.path.exists(self.checkp_dir): os.makedirs(self.checkp_dir)


    def set_epoch_step(self, epoch, step):
        '''Set epoch/step state of the logger for appropriate event handling'''
        self.curr_epoch = epoch 
        self.curr_step  = step
        with open(self.state_file, 'a') as out_file:
            out_file.write(f'\n  Epoch {epoch}, Step {step}\n')


    def vis_save(self, *args, obey_intervals=True, **kwargs):
        '''Save the visualization per the vis_save_f function. 
            If obeying intervals, only do it when state is appropriate
        '''
        if obey_intervals and (self.curr_epoch % self.vis_interv): return
        curr_file = f'{self.model_name}_e{self.curr_epoch:04d}_s{self.curr_step:04d}'
        self.vis_save_f(*args, **kwargs, 
            epoch = self.curr_epoch, 
            step = self.curr_step, 
            out_path = f'{self.picout_dir}/{curr_file}'
        )


    def write_loss(self, losses):
        '''Write loss statistics to the loss file. Flushes to new handler'''
        with open(self.loss_file, 'a') as out_file:
            out_file.write(f'Epoch {self.curr_epoch}: ')
            out_file.write(' | '.join([f'{k} : {v}' for k,v in losses]))
            out_file.write('\n')


    def log(self, msg):
        '''Log a timestamped message to the log file. Flushes to new handler'''
        with open(self.state_file, 'a') as out_file:
            out_file.write(f' > {time.strftime("%H:%M:%S", time.localtime())} | {msg}\n')


    def load_cp(self, models, opt):
        '''Load checkpoints (by reference) from the specified file directory'''
        try:
            with open(f'{self.meta_file}', 'r') as in_file: 
                n_models = int(in_file.readline())        
                model_names = [in_file.readline().rstrip() for _ in range(n_models)]
                for i, name in enumerate(model_names): 
                    self.checkpoints.append(tf.train.Checkpoint(optimizer=opt, model=models[i]))
                    self.checkp_mgrs.append(tf.train.CheckpointManager(
                        self.checkpoints[-1], f'{self.checkp_dir}/{name}', max_to_keep=3))
                [cp.restore(mgr.latest_checkpoint) for cp, mgr in zip(self.checkpoints, self.checkp_mgrs)]
                curr_epoch = int(in_file.readline())
                return curr_epoch
        except Exception as e: 
            print(f"Could not load '{self.model_name}'-associated checkpoints: \n\t{e}")
            return 0


    def save_cp(self, models, names, opt, obey_intervals=True):
        '''Save the checkpoint with models, names, and optimizers. 
            If obeying intervals, only do it when state is appropriate
        '''
        # Skip the save_cp if interval == 0, curr step == 0, or the epoch does not conform to interval
        if obey_intervals and (not self.chp_interv or self.curr_step or (self.curr_epoch - 1) % self.chp_interv): return
        with open(f'{self.meta_file}', 'w') as out_file: 
            out_file.write(f'{len(models)}\n')
            for i, (name, model) in enumerate(zip(names, models)):
                out_name = f'{self.model_name}_{name}'
                if i >= len(self.checkpoints):
                    self.checkpoints.append(tf.train.Checkpoint(optimizer=opt, model=model))
                    self.checkp_mgrs.append(tf.train.CheckpointManager(
                        self.checkpoints[-1], f'{self.checkp_dir}/{out_name}', max_to_keep=3))
                self.checkp_mgrs[i].save()
                out_file.write(f'{out_name}\n')
            out_file.write(f'{self.curr_epoch}\n')
