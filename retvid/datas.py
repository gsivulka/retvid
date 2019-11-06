import os
from itertools import repeat
import torch
import skimage
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as vistrans
import torchdeepretina as tdr

# Prevents io error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class StimGen(Dataset):
    def __init__(self, n_samples=10000, filt_depth=40):
        self.n_samples = 10000
        self.filt_depth = filt_depth
        self.gen_fxns = [self.contrast_adaptation, self.motion_anticipation, 
                                self.motion_reversal, self.oms, self.osr, 
                                self.reversing_grating, self.step_response]
        self.n_classes = len(self.gen_fxns)

    def __len__(self):
        return self.n_samples

    def contrast_adaptation(self, c0=1.75, c1=0.25, tot_dur=150):
        """
        Step change in contrast

        c0: float
            a value for the contrast. the larger of the two contrasts
        c1: float
            a value for the contrast. the smaller of the two contrasts
        tot_dur: int
            number frames (assume 10ms/frame for model training)
        """
        # the contrast envelope
        qrtr_dur = int((tot_dur-self.filt_depth)//4)
        remainder = (tot_dur-self.filt_depth)%4
        flicker_1 = tdr.stimuli.repeat_white(self.filt_depth+qrtr_dur, nx=1, contrast=c0,
                                                                         n_repeats=1)
        flicker_2 = tdr.stimuli.repeat_white(qrtr_dur*2, nx=1, contrast=c1, n_repeats=1)
        flicker_3 = tdr.stimuli.repeat_white(qrtr_dur+remainder, nx=1, contrast=c0, n_repeats=1)
        envelope = np.concatenate([flicker_1, flicker_2, flicker_3], axis=0)
        envelope = tdr.stimuli.spatialize(envelope, nx=50)
        return envelope

    def motion_anticipation(self, velocity=0.08, width=2):
        """
        Generates the Berry motion anticipation stimulus

        velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as Berry et. al.
        width = 2               # 2 bars == 110 microns, Berry et. al. used 133 microns

        Returns
        -------
        stim : array_like
        """
        heads = np.random.random()>.5

        # moving bar stimulus and responses
        # c_right and c_left are the center positions of the bar
        if heads:
            c_left, speed_left, stim_left = tdr.stimuli.driftingbar(-velocity, width, x=(7, -8))
            stim_left = stim_left[:,-1]
            stim_left = np.concatenate([stim_left,stim_left[-1:], stim_left[-1:]], axis=0)
            stim = stim_left
        else:
            c_right, speed_right, stim_right = tdr.stimuli.driftingbar(velocity, width,
                                                                              x=(-7, 8))
            stim_right = stim_right[:,-1]
            stim_right = np.concatenate([stim_right,stim_right[-1:], stim_right[-1:]], axis=0)
            stim = stim_right
        std = 0.2
        return stim + std*np.random.randn(*stim.shape)

    def motion_reversal(self, velocity=0.08, width=2):
        """
        Moves a bar to the right and reverses it in the center, then does the same to the left. 
        The responses are averaged.
        Parameters
        ----------
        velocity = 0.08         # 0.08 bars/frame == 0.44mm/s, same as Berry et. al.
        width = 2               # 2 bars == 110 microns, Berry et. al. used 133 microns

        Returns
        -------
        stim : array_like
        """
        # moving bar stimuli
        c_right, speed_right, stim_right = tdr.stimuli.driftingbar(velocity, width, x=(-8,8))
        stim_right = stim_right[:,-1]
        c_left, speed_left, stim_left = tdr.stimuli.driftingbar(-velocity, width, x=(8, -8))
        stim_left = stim_left[:,-1]

        # Find point that bars are at center
        right_halfway = None
        left_halfway = None 
        half_idx = stim_right.shape[1]//2
        for i in range(len(stim_right)):
            if right_halfway is None and stim_right[i,0, half_idx] <= -.99:
                right_halfway = i
            if left_halfway is None and stim_left[i, 0, half_idx] <= -.99:
                left_halfway = i
            if right_halfway is not None and left_halfway is not None:
                break
        # Create stimulus from moving bars
        rtl = np.concatenate([stim_right[:right_halfway], stim_left[left_halfway:]], axis=0)
        ltr = np.concatenate([stim_left[:left_halfway], stim_right[right_halfway:]], axis=0)
        if right_halfway < left_halfway:
            cutoff = left_halfway-right_halfway
            ltr = ltr[cutoff:-cutoff]
        elif left_halfway < right_halfway:
            cutoff = right_halfway-left_halfway
            rtl = rtl[cutoff:-cutoff]

        heads = np.random.random()>.5
        if heads:
            stim = np.concatenate([rtl, rtl[-1:], rtl[-1:], rtl[-1:]], axis=0)
        else:
            stim = np.concatenate([ltr, ltr[-1:], ltr[-1:], ltr[-1:]], axis=0)
        std = 0.2
        return stim + std*np.random.randn(*stim.shape)

    def oms(self, tot_frames=150, pre_frames=40, post_frames=40, img_shape=(50,50), 
                                                        center=(25,25), radius=8, 
                                                        background_velocity=.3, 
                                                        foreground_velocity=.5, 
                                                        seed=None, bar_size=2, 
                                                        inner_bar_size=None):
        """
        Plays a video of differential motion by keeping a circular window fixed in space on a 
        2d background grating.
        A grating exists behind the circular window that moves counter to the background
        grating. Each grating is jittered randomly.
    
        tot_frames: int
            total length of video in frames
        pre_frames: int
            number of frames of still image to be prepended to the jittering
        post_frames: int
            number of frames of still image to be appended to the jittering
        img_shape: sequence of ints len 2
            the image size (H,W)
        center: sequence of ints len 2
            the starting pixel coordinates of the circular window (0,0 is the upper left most
            pixel)
        radius: float
            the radius of the circular window
        background_velocity: float
            the intensity of the horizontal jittering of the background grating
        foreground_velocity: float
            the intensity of the horizontal jittering of the foreground grating
        seed: int or None
            sets the numpy random seed if int
        bar_size: int
            size of stripes. Min value is 3
        inner_bar_size: int
            size of grating bars inside circle. If None, set to bar_size
        """
        if seed is not None:
            np.random.seed(seed)
        diff_frames = int(tot_frames-pre_frames-post_frames)
        assert diff_frames > 0
        differential, _, _ = tdr.stimuli.random_differential_circle(diff_frames, 
                                        bar_size=bar_size, 
                                        inner_bar_size=inner_bar_size,
                                        foreground_velocity=foreground_velocity, 
                                        background_velocity=background_velocity,
                                        image_shape=img_shape, center=center, radius=radius) 
        pre_vid = np.repeat(differential[:1], pre_frames, axis=0)
        post_vid = np.repeat(differential[-1:], post_frames, axis=0)
        stim = np.concatenate([pre_vid, differential, post_vid], axis=0)
        std = 0.2
        return stim + std*np.random.randn(*stim.shape)

    def osr(self, duration=2, interval=6, nflashes=8, intensity=-2.0, noise_std=0.2):
        """Omitted stimulus response
    
        Parameters
        ----------
        duration : int
            Length of each flash in samples (default: 2)
    
        interval : int
            Number of frames between each flash (default: 10)
    
        nflashes : int
            Number of flashes to show before the omitted flash (default: 5)
    
        intensity : float
            The intensity (luminance) of each flash (default: -2.0)
        """
    
        # generate the stimulus
        single_flash = tdr.stimuli.flash(duration, interval, duration+interval,
                                                            intensity=intensity)
        # Adding noise
        single_flash = single_flash + noise_std*np.random.randn(*single_flash.shape)
        omitted_flash = tdr.stimuli.flash(duration, interval, duration+interval, intensity=0.0)
        # Adding noise
        omitted_flash = omitted_flash + noise_std*np.random.randn(*omitted_flash.shape)
        flash_group = list(repeat(single_flash, nflashes))
        zero_pad = np.zeros((40-interval, 1, 1))
        start_pad = np.zeros((interval * (nflashes-1), 1, 1))
        X = tdr.stimuli.concat(start_pad, zero_pad, *flash_group, omitted_flash, zero_pad,
                                                                            nx=50, nh=40)
        X = X[:,-1]
        padding = np.zeros((8,50,50))
        X = np.concatenate([padding,X], axis=0)
        return X

    def reversing_grating(self, size=5, phase=0.):
        """
        A reversing grating stimulus

        size: int
            width of the bar
        phase: float
            ???
        """
        grating = tdr.stimuli.grating(barsize=(size, 0), phase=(phase, 0.0),
                                                        intensity=(1.0, 1.0),
                                                        us_factor=1, blur=0)
        grating = tdr.stimuli.reverse(grating, halfperiod=50, nsamples=150)
        std = 0.2
        grating = grating + std*np.random.randn(*grating.shape)
        return grating

    def step_response(self, flash_dur=60, delay=50, nsamples=150, intensity=-1):
        std = 0.1
        tempensity = intensity + std*np.random.randn(1)[0]
        flash = tdr.stimuli.flash(flash_dur, delay, nsamples, intensity=tempensity)
        flash = tdr.stimuli.spatialize(flash, 50)
        noise = std*np.random.randn(*flash.shape)
        flash = flash + noise
        return flash

    def __getitem__(self, idx):
        mod_idx = idx % self.n_classes
        print("mod_idx", mod_idx)
        gen_fxn = self.gen_fxns[mod_idx]
        stim = gen_fxn()
        stim = tdr.utils.rolling_window(stim, self.filt_depth) 
        x = torch.FloatArray(stim)
        y = mod_idx
        return x, y # (T, D, H, W)


def get_data_split(dataset, filt_depth=40):
    """
    Returns two torch Datasets, one validation and one training.

    dataset: str
        the name of the desired dataset
    """

    train_set = StimGen(n_samples=100000, filt_depth=filt_depth)
    val_set = StimGen(n_samples=10000, filt_depth=filt_depth)
