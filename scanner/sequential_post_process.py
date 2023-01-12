# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:34:08 2022

@author: gabri
"""

import sys
import os
from pathlib import Path
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as io
from scipy.signal import windows, resample, chirp

# Pytta imports
import pytta
#from pytta.generate import sweep
# from pytta.classes import SignalObj, FRFMeasure
# from pytta import ImpulsiveResponse, save, merge


# Receiver class
from receivers import Receiver


# Field decomposition imports
from controlsair import AlgControls, AirProperties, load_cfg
from zs_array_estimation import ZsArray

# utils
import utils



class ScannerPostProcess():
    
    
    
    def __init__(self, main_folder = 'D:', name = 'name', material='Sonex',            
            fs = 44100, fft_degree = 18, repetitions = 2, t_bypass = 0.0, temp = 27.2,
            start_stop_margin = [0.1, 3.5], mic_sens = None):
        
        # folder checking
        self.main_folder = Path(main_folder)
        self.name = name
        self.check_main_folder()
        self.material = material
        # audio signals checking
        self.fs = fs
        self.fft_degree = fft_degree
        self.start_margin = start_stop_margin[0]
        self.stop_margin = start_stop_margin[1]
        #self.micro_steps = 1600
        self.mic_sens = mic_sens
        self.repetitions = repetitions
        self.t_bypass = t_bypass
        #
        self.xt = []
        self.temp = temp
        self.IRs_windowed = []
        self.f_Decomp = []
        self.pf_Decomp = []
        
    def check_main_folder(self,):
        folder_to_test = self.main_folder / self.name
        if folder_to_test.exists():
            print('Measurement path found! Use this object to read only.')
        else:
            folder_to_test.mkdir(parents = False, exist_ok = False)
            measured_signals_folder = folder_to_test / 'measured_signals'
            measured_signals_folder.mkdir(parents = False, exist_ok = False)
            
    def set_meas_sweep(self, method = 'logarithmic', freq_min = 1,
                       freq_max = None, n_zeros_pad = 0):
        """Set the input signal object
        
        The input signal is called "xt". This is to ease further
        implementation. For example, if you want to set a random signal,
        you can also call it "xt", and pass it to the same method that 
        computes the IR
        
        Parameters
        ----------
        method : str
            method of sweep
        freq_min : float
            minimum frequency of the sweep
        freq_max : float or None
            maximum frequency of the sweep. Default is None, which sets
            it to fs/2
        """
        self.freq_min = freq_min
        if freq_max is None:
            self.freq_max = int(self.fs/2)
        else:
            self.freq_max = freq_max
        
        self.method = method
        self.n_zeros_pad = n_zeros_pad
        
        # set pytta sweep
        xt = pytta.generate.sweep(freqMin = self.freq_min,
          freqMax = self.freq_max, samplingRate = self.fs,
          fftDegree = self.fft_degree, startMargin = self.start_margin,
          stopMargin = self.stop_margin,
          method = self.method, windowing='hann')
        new_xt = np.zeros(len(xt.timeSignal[:,0]) + n_zeros_pad)
        new_xt[:len(xt.timeSignal[:,0])] = xt.timeSignal[:,0]
        
        # time = np.linspace(0, (2**self.fft_degree-1)/self.fs, 2**self.fft_degree)
        # xt = chirp(t = time, f0 = self.freq_min, t1 = time[-1], f1 = self.freq_max,
        #            method = 'logarithmic')
        # new_xt = np.zeros(len(xt) + n_zeros_pad)
        # new_xt[:len(xt)] = xt
        
        
        self.xt = pytta.classes.SignalObj(
            signalArray = new_xt, 
            domain='time', samplingRate = self.fs)
    
    def pytta_play_rec_setup(self,):
        """ Configure measurement of response signal using pytta and sound card
        """
        self.pytta_meas = pytta.generate.measurement('playrec',
            excitation = self.xt,
            samplingRate = self.fs,
            freqMin = 1,
            freqMax = self.fs/2,
            device = (9,9),
            inChannels=[1],
            outChannels=[1])
    
    def array_config(self, Type = 'double planar array', Lx = 0.57, Ly = 0.65, nx = 11, ny = 12,
                                                 zr = 0.013, dz = 0.029, pt0 = np.array([0.0, 0.0, 0.01]),
                                                 source_coord = np.array([0.0, 0.0, 1.09])):
        """
        Parameters
        ----------
        Type : STR, optional
            Type of the array setup. The default is 'double planar array'.
        Lx : FLOAT, optional
            X-range of array. The default is 0.57.
        Ly : FLOAT, optional
            Y-range of array. The default is 0.65.
        nx : INT, optional
            Number of points in X-range. The default is 11.
        ny : INT, optional
            Number of points in Y-range. The default is 12.
        zr : FLOAT, optional
            Distance between the lower plane of the array and the material's surface.
            The default is 0.013.
        dz : FLOAT, optional
            Distance between the array planes. The default is 0.029.
        pt0 : ARRAY OF FLOAT
            Location of the microphone manually positioned before the measurement.
            The default is np.array([0.0, 0.0, 0.01]).
        source_coord : ARRAY OF FLOAT
            Location of the source in the measurement scenario.
            The default is np.array([0.0, 0.0, 1.09]).
        """       
        def _euclid_dist(r1,r2):
            d = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2])**2)
            return d
        
        self.pt0 = pt0
        self.source_coord = source_coord
        
        receiverObj = Receiver()
        if Type == 'double planar array':
            receiverObj.double_planar_array(x_len=0.57,n_x=11,y_len=0.65,n_y=12, zr=0.013, 
                                               dz=0.029)
            # original array
            self.receivers = receiverObj
            # Changing order of the points:
            order1 = utils.order_closest(pt0, self.receivers.coord)
                # new array
            self.receivers.coord = order1
            self.source_rec_euclidian_distances = np.zeros((max(order1.shape), ), dtype='float64')
            for pt in range(max(order1.shape)):
                self.source_rec_euclidian_distances[pt] = _euclid_dist(source_coord,order1[pt])
                # creating the matrix with all distances between all points
           # self.stand_array = utils.matrix_stepper(self.pt0, self.receivers.coord)
        else:
            pass
            
        
            
    
    def load(self,):
        """Loads the measurement control object as pickle

        """
        pickle_name = self.name + '.pkl'
        path_filename = self.main_folder / self.name / pickle_name
        with path_filename.open(mode = 'rb') as f:
            tmp_dict = pickle.load(f)
        f.close()
        self.xt = pytta.generate.sweep(freqMin = 1, freqMax = self.fs/2, samplingRate= self.fs,
                                       startMargin=self.start_margin, stopMargin=self.stop_margin,
                                       method = 'logarithmic')
        self.pytta_play_rec_setup()
        self.__dict__.update(tmp_dict)        
    
    
    def load_meas_files(self,):
        """Load all measurement files
        """
        yt_list = []
        for jrec in range(self.receivers.coord.shape[0]):
            y_rep_list = []
            for jmeas in range(self.repetitions):
                filename = 'rec' + str(int(jrec)) +\
                        '_m' + str(int(jmeas)) + '.hdf5'
                complete_path = self.main_folder / self.name / 'measured_signals'
                med_dict = pytta.load(str(complete_path / filename))
               # new_dict = {'repetitions':str(int(jmeas))}
               # med_dict.update(new_dict)
                y_rep_list.append(med_dict['plot_frf'])
            yt_list.append(y_rep_list)
            
        return yt_list
    
    def load_and_processIR_meas_files(self, kirkRegularization = True, discout_bypass_time=True, 
                                      store_as_objs = True):
        """
        Load audio measurements along with it's IR process 
        
        Parameters
        ----------
        kirkRegularization : "True" or "False"
            Apply or not Kirkeby's regularization. Default is 'True'.
        discout_bypass_time:  "True" or "False"
            After the time average, the signal is cropped to t == [t_bypass : 1/fs : end]. 
            Default is 'True'
        store_as_objs :  "True" or "False" 
            When is set to 'True', all IR and storaged in the class as 'IRs'. When lots of point are processed 
            inside a loop, it will consume a lot more memory. By setting this to 'False', only the time signal
            of the IR will be storaged.
        """
        self.IR_pts = []
        for jrec in range(self.receivers.coord.shape[0]):
           # y_rep_list = []
            IR_rep_list = []
            for jmeas in range(self.repetitions):
                filename = 'rec' + str(int(jrec)) +\
                        '_m' + str(int(jmeas)) + '.hdf5'
                complete_path = self.main_folder / self.name / 'measured_signals'
                med_dict = pytta.load(str(complete_path / filename))
                IR_tk = pytta.ImpulsiveResponse(excitation=self.xt, recording=med_dict['plot_frf'], samplingRate=self.fs, regularization = kirkRegularization) 
                # new_dict = {'repetitions':str(int(jmeas))}
                # med_dict.update(new_dict)
                IR_rep_list.append(IR_tk.IR)
            if self.repetitions == 2:
                IR_pt_merge = pytta.merge(IR_rep_list[0], IR_rep_list[1])
            elif self.repetitions == 3:
                IR_pt_merge = pytta.merge(IR_rep_list[0], IR_rep_list[1], IR_rep_list[2])
                # y_rep_list.append(med_dict['plot_frf'])
            IRpt = IR_pt_merge.channelMean() 
            IRpt.comment = self.receivers.coord[jrec]
            if discout_bypass_time == True:
                IRpt.crop(float(self.t_bypass), float(IRpt.timeVector[-1]))
            
            if store_as_objs == True:
                self.IR_pts.append(IRpt)
            else:
                self.IR_pts.append(IRpt.timeSignal)
            #yt_list.append(y_rep_list)
        self.t_IR = IRpt.timeVector    
        
    def temp_average(self, Yt = [[]], rep = 2, kirkRegularization = True,
                     discout_bypass_time=True):
        """
        Impulsive Response's temporal average of a point.
        
        Parameters
        ----------
        Yt : "list" 
            Each element contains a 'rep' number of signal objects which were
            taken on one point
        kirkRegularization : "True" or "False"
            Apply or not Kirkeby's regularization. Default is 'True'.
        discout_bypass_time:  "True" or "False"
            After the time average, the signal is cropped to t == [t_bypass : 1/fs : end]. 
            Default is 'True'
        -------
        
        Returns
        -------
        IRs_array : "list" - Each element contains the time signal vector result of each point

        """
        rep = self.repetitions
        IRs_array = []
        #rep=2
        for r in range(self.receivers.coord.shape[0]):
            if rep > 1:
                tk=[] 
                for rr in range(rep):
                    ht_obj = pytta.ImpulsiveResponse(excitation=self.xt, recording=Yt[r][rr], samplingRate=self.fs, regularization = kirkRegularization)
                    tk.append(ht_obj)
                hts_pt = pytta.merge(tk[0].IR, tk[1].IR)       
                ht = hts_pt.channelMean()
                ht.comment = self.receivers.coord[r]
                if discout_bypass_time == True:
                    ht.crop(float(self.t_bypass), float(ht.timeVector[-1]))
                IRs_array.append(ht)
            else:
                ht_obj = pytta.ImpulsiveResponse(excitation=self.xt, recording=Yt[r][0], samplingRate=self.fs, regularization = kirkRegularization)
                ht_pt = ht_obj.IR
                ht_pt.comment = self.receivers.coord[r]
                if discout_bypass_time == True:
                    ht_pt.crop(float(self.t_bypass), float(ht_pt.timeVector[-1]))
                IRs_array.append(ht_pt)
                
        return IRs_array
        
    def temp_window(self, Imp_Resp = [], s_coord=np.array([0.0, 0.0, 1.0]), r_coord = np.array([0.0, 0.0, 0.01]),
                    Exp_params=True, baltParams=False, 
                    tw1=0.8, tw2=2.5, T3 = 0.0094, t_start=0.00002,    #Window params
                    blackmanharris_or_hann = 'blackman harris', 
                    plot=False, savefig=False, path= '', name = ''):
           """
           Adrienne's window creation and application. The window is made by two halfs of the Blackman Harris window, an ascending
           plus a descending part. The ascending part represents the beggining of the Adrienne's window, which follows a retangular
           (flat) window (where it must contain the record of the IR's direct incidence), and a descending part of the Blackman-Harris
           window represents the end of the Adrienne's window.
           
           Parameters
           ----------
           Imp_Resp : [float/object] 
               Impulsive resposte time vector or it's signal object 
           s_coord : [list/array]
               Coordinates of source position
           r_coord : [list/array]
               Coordinates of receiver point 
           Exp_params : "True" or "False" 
               Experimental detection of the direct incidence time arrival. Default is "True"
           baltParams : 
               When is set to "True", the window parameters are calculated by the same way as it is in the article "European methodology 
               for testing the airborne sound insulation characteristics of noise barriers in situ Experimental verification and comparison
               with laboratory data", in which:
                   - T1 timelength is 0,5 ms
                   - T2 timelength is 5,18 ms
                   - T3 length is 2,22 ms
           t_start : [float]
               Instance when the window's ascending part starts
           tw1 : 0 < [float] < 1:
               Ascending part of the window goes up to 'tw1*td'
           tw2 : 1 < [float]:
               Descending part of the window starts on 'tw2*td'
           T3 : [float]:
               Timelength of the window's descending part 
           blackmanharris_or_hann : "blackman harris" or "hann"
               Type of the window used to generate the ascending and descending part of temporal window
           IRcheck : [array/list]
               Array of the points which you want to plot. Also, the 'plot' parameter
               must be set to "True"
           path : [string]
               Path string of location to save the windowed IR plot. Also, the 'save' parameter
               must be set to "True"
           name : [string]
               Name of the figure which you want to save it.
               
           Returns
           -------
           IR_windowed : "array of float64" - time signal of the windowed impulsive response 
           window : "array of float 64" - window's vector

           """
           
           if type(Imp_Resp) != np.ndarray:
               Pt = Imp_Resp.timeSignal
               t_Pt = Imp_Resp.timeVector
           elif type(Imp_Resp) == list:
               if type(Imp_Resp[0]) != np.ndarray:
                   Pt = Imp_Resp.timeSignal
                   t_Pt = Imp_Resp.timeVector
               else:
                   Pt = Imp_Resp[0]
           else:
               Pt = Imp_Resp
               t_Pt = np.linspace(0, len(Pt)/self.fs-1/self.fs,len(Pt), dtype='float64')
           # Experimental identification of the direct arrival time (if selected):
           if Exp_params is True:
               Max = max(Pt); Max_idx = np.where(Pt==Max)
               td = float(Max_idx[0]/self.fs)
           else:
               # Theoretical arrival time - Euclidian distance between microhpone and source's position:
               d = np.sqrt((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2 + (s_coord[2] - r_coord[2])**2)
               td = np.divide(d, 331.3*np.sqrt(1 + np.divide(self.temp, 273.15)))     
           if baltParams is True:
               tw1 = 1 - 0.0002/td
               t_start = float(tw1*td - 0.0005)
               if t_start < 0:
                   print('\n "t_start" is negative! Increase "tw1" value! \n')
               tw2 = tw1 + 0.00518/td
               t_end = tw2*td + 0.00222
               T1 = round(tw1*td - t_start,5); T2 = round(td*(tw2 - tw1),5); T3 = round(t_end - tw2*td,5)
               w0 = np.zeros(round(t_start*self.fs));            W0 = w0
           else:
               T1 = tw1*td - t_start;      
               T2 = (tw2 - tw1)*td;  
               t_end = T3 + tw2*td
               #T3 = t_end - tw2*td;       
               w0 = np.zeros(round(t_start*self.fs)); W0 = w0  
           if blackmanharris_or_hann == 'blackman harris':
                   w1 = windows.blackmanharris(int(2*T1*self.fs));   W1 = w1[:len(w1)//2]
                   w2 = windows.boxcar(int(T2*self.fs));             W2 = w2
                   w3 = windows.blackmanharris(int(2*T3*self.fs));   W3 = w3[len(w3)//2:]
                   adri_wind = np.concatenate((W1, W2, W3))
                   wind = np.concatenate((W0, adri_wind))
           else:
                   w1 = windows.hann(int(2*T1*self.fs));   W1 = w1[:len(w1)//2]
                   w2 = windows.boxcar(int(T2*self.fs));   W2 = w2
                   w3 = windows.hann(int(2*T3*self.fs));   W3 = w3[len(w3)//2:]
                   adri_wind = np.concatenate((W1, W2, W3))
                   wind = np.concatenate((W0, adri_wind))
           #t_wind = np.linspace(0, round(wind.shape[0]/self.fs, 5), wind.shape[0])
           delta_t_samples = max(Pt.shape) - int(len(wind))
           end_zeros = np.zeros(int(delta_t_samples))
           window = np.concatenate((wind, end_zeros))
           t_window = np.linspace(0, round(max(window.shape)/self.fs, 5), max(window.shape))
           "Applying the window:"
           IR_windowed = np.zeros((max(Pt.shape),), dtype='float64')
           for j in range(len(window)):
               IR_windowed[j] = np.multiply(window[j], Pt[j])
           if plot is True:
               plt.figure()
               plt.grid(color='gray', linestyle='-.', linewidth=0.4)
               plt.title('Resp. Impulsiva + janela temporal')
               plt.plot(t_window, Pt/max(Pt), linewidth=1.5, label='RI')
               plt.plot(t_window, window, linewidth=2.8, label='Janela')
               plt.xlim((0.0, 0.035));   plt.ylim((-0.9, 1.1))
               plt.xticks([0.0, 0.003, 0.006, 0.009, 0.0120, 0.015, 0.018], 
                          ['0,0', '3,0', '6,0', '9,0', '12,0', '15,0', '18,0'])
               plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75, 1.0], 
                          ['-0,75','-0,5', '-0.25', '0,0', '0,25', '0,50', '0,75', '1,00'])
               plt.legend(loc='best', fontsize='medium')
               plt.xlabel('Tempo [ms]')
               plt.ylabel('Amplitude normalizada')
               plt.tight_layout()
               plt.show()
               if savefig is True:
                   nameFig = path + name + '.png'
                   plt.savefig(nameFig, dpi=300)
           self.IRs_windowed.append(IR_windowed)
           return IR_windowed, window  
        
    
    

    
    
    
    
    
    def time_and_freq_crop_to_decomp(self, t_crop = 0.2, freq_range = np.array([100, 4000]),
                                     return_freq_and_pf = False):
        """
        After windowing the signals, they're full of zeros after the reflected component detection.
        You will take too much time to decompose the field using the full resolution of the signals.
        In this way, this function is used to crop them. By doing that, it also reduces the spectrum
        resolution - i.e. for t_crop = 0.2, the frequency resolution will be 10 Hz.

        Parameters
        ----------
        t_crop : FLOAT, optional
            Time value to crop the signal. The default is 0.2.
        freq_range : TYPE, optional
            Range of the frequency vector which will be used in the field decomposition.
            The default is [100, 4000].
        return_freq_and_pf : "True" or "False", optional
            In any way, the frequency and the pressures vector will be attached to the class.
            But, if you need for the function to return pf and f to variables of your main code,
            set this parameter to 'True'. The default is 'False'.

        Returns
        -------
        f_Decomp : ARRAY of FLOAT64
            Frequency vector for decompositions.
        pfDecomp : 2D-ARRAY of FLOAT64
            Array with the spectrums of all points for decomposition.

        """  
        pfDecomp_Lst = []
        for i in range(self.receivers.coord.shape[0]):
              ptObj = pytta.classes.SignalObj(signalArray = self.IRs_windowed[i],
                                              domain = 'time', freqMin=1, freqMax=self.fs/2, samplingRate = self.fs) 
              ptObj.crop(0, t_crop)
              pf_Dec = ptObj.freqSignal[:,0]
              pfDecomp_Lst.append(pf_Dec)
              if i==1:
                  f_Decomp = ptObj.freqVector 
              else:
                  pass

        f_minIDX = int(np.where(f_Decomp==freq_range[0])[0])
        f_maxIDX = int(np.where(f_Decomp==freq_range[1])[0])

        f_Decomp = f_Decomp[f_minIDX:f_maxIDX] # Selecting frequencies up to max(freq_range)
        pfDecomp = np.zeros((len(pfDecomp_Lst), len(f_Decomp)), dtype = 'complex64')
        for i in range(len(pfDecomp_Lst)):
            pfDecomp[i,:] = pfDecomp_Lst[i][f_minIDX:f_maxIDX]
        
        self.freq_decomp = f_Decomp
        self.pf_decomp = pfDecomp 
        if return_freq_and_pf == True:
            return f_Decomp, pfDecomp
        
    def array_rearrange(self, Lxy_material = np.array([0.41, 0.41])):
        
        
        max_Lx_array = round(max(self.receivers.coord[:,0]),5); min_Lx_array = round(min(self.receivers.coord[:,0]),5)
        max_Ly_array = round(max(self.receivers.coord[:,1]),5); min_Ly_array = round(min(self.receivers.coord[:,1]),5)
   
    def field_decomposition(self, n_waves = 2542, Method = 'Ridge', plot_spheres = False, plot_maps = False,
                            f_plot = np.array([100, 250, 500, 1000, 1500, 2000, 2500, 300, 3500, 4000]), 
                            dB_range = 30, save_plots = False, save_field = False, path = ''):
        """
        Function used to decompose the sound field. The results will be attached to the class

        Parameters
        ----------
        n_waves : INT, optional
            Number of plane-wave components of the spectrum. The default is 2542.
        Method : STR, optional
            Mathematical method of the decomposition. The default is 'Ridge'.
        plot_spheres : TRUE or FALSE, optional
            Either or not to plot the sphere's wavenumber spectrums. The default is False.
        plot_maps : TRUE or FALSE, optional
            Either or not to plot the 2D's wavenumber spectrums. The default is True.
        f_plot : ARRAY, optional
            Frequencies which you want to plot the spectrums. The default is np.array([100, 250, 500, 600, 1000, 1240, 2000, 4000]).
        dB_range : INT or FLOAT, optional
            Decibel range of the plots. The default is 20.
        save_plots : TRUE or FALSE, optional
            Either save or not the spectrums plotted. The default is False.
        save_field : TRUE or FALSE, optional
            Either save or not the field decomposition object with it's results. The default is False.
        path : STR, optional
            Path of location to save if 'save_field' is 'True'.

        Returns
        -------
        None.

        """
        
        air = AirProperties(temperature = self.temp)
        controls = AlgControls(c0 = air.c0, freq_vec = self.freq_decomp) 
        
        self.field = ZsArray(p_mtx=self.pf_decomp, controls=controls, receivers = self.receivers)
        self.field.wavenum_dir(n_waves=n_waves, plot = False)
        self.field.pk_tikhonov(method = Method, plot_l = False)
        self.field.pk_interpolate()
        
        db_range = dB_range
        if plot_spheres == True:
            fName = 'pk_'+self.material + '_' 
            pkm = path + self.material + '/class_results/'; pks = pkm
            for f in range(len(f_plot)):    
                if save_plots == True:     
                    self.field.plot_pk_sphere2(freq=f_plot[f], db=True, dinrange=db_range,
                                          save=True, name=fName, path=pks, travel=False)
                else:
                   self.field.plot_pk_sphere2(freq=f_plot[f], db=True, dinrange=db_range,
                                         save=False, name=fName, path=pks, travel=False) 
                if save_plots == True:
                    self.field.plot_pk_sphere2(freq=f_plot[f], db=True, dinrange=db_range, save=False,
                                      name=fName, path=self.main_folder, travel=False)
                else:
                    self.field.plot_pk_sphere2(freq=f_plot[f], db=True, dinrange=db_range, save=False,
                                      name=fName, path=self.main_folder, travel=False)
        else:
            pass 
        if plot_maps == True:
            fName = 'pk_'+self.material + '_' 
            pkm = path + self.material + '/class_results/'; pks = pkm
            for f in range(len(f_plot)):
                if save_plots == True:
                    self.field.plot_pk_smallmap(freq=f_plot[f], db=True, dinrange=db_range, 
                                            title = False, colorbar=False, save=True, fname='Pk_Smap_'+fName, path=pkm)
                else:
                    self.field.plot_pk_smallmap(freq=f_plot[f], db=True, dinrange=db_range, 
                                            title = False, colorbar=False, save=False, fname='Pk_Smap_'+fName, path=pkm)
        else:
            pass
        
        if save_field == True:
            self.field.save(filename='Field_decomposition', path = path + self.material + '/class_results/')
            
            
    def reconstruct_pu(self, z_coord = 0.0, Lxy = np.array([0.1, 0.1]), 
                                   nxy = np.array([21, 21], dtype='int32'), theta_rad = 0,
                                   alpha_from_wavenumber = True, plot_alphas = False):
        """
        Function used to reconstruct the pressure and particle velocity outside the area scanned
        by the array. With 'p' and 'u', the surface impedance is calculated, also the absorption
        coefficient is reconstructed.

        Parameters
        ----------
        z_coord : FLOAT, in meters
            The height of the points where pressure and particle velocity will be
            estimated. The default is 0.0.
        Lxy : [FLOAT, FLOAT], in meters
            XY-range of the plane of points where 'p' ans 'u' will be estimated.
            The default is np.array([0.1, 0.1]).
        nxy : [INT, INT], optional
            Number of points integrating the X and Y range of the plane, respectively.
            The default is np.array([21, 21]).
        theta_rad : FLOAT, in radians
            Incidence angle of scenario's measurement. The default is 0 (normal incidence).
        alpha_from_wavenumber : TRUE or FALSE
            Either or not to (also) estimate the absorption coefficient directly from the
            wavenumber spectrum components. The default is True.
        plot_alphas : TRUE or FALSE
            Either or not to plot the results. The default is False.

        Returns
        -------
        None.

        """
        alphas = []; freq_plot = []
        self.field.zs(Lx=0.1, Ly=0.1, n_x=21, n_y=21, zr=0.0, theta=[np.deg2rad(0)])  
        alpha = self.field.alpha[0,:] 
        alphas.append(alpha); freq_plot.append(self.freq_decomp)
        if alpha_from_wavenumber != False:
            self.field.alpha_from_pk()
            alphas.append(self.field.alpha_pk)
            freq_plot.append(self.freq_decomp)
        
        if plot_alphas == True:
            if alpha_from_wavenumber != False:
                leg=['rec.',r'from $P(k)$']
                plt.figure()
                plt.title('Absorption coefficients reconstructed')
                for al in range(2):
                    plt.semilogx(freq_plot[al], alphas[al], label=leg[al], linewidth = 2.0)   
            else: 
                plt.figure()
                plt.title('Aborption coefficient reconstructed')
                plt.semilogx(freq_plot[0], alphas[0], label='rec.', linewidth = 2.0)             
            plt.xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            plt.ylim((-0.03, 1.02))
            plt.xlabel('Frequency [Hz]', fontsize='medium')
            plt.ylabel(r'$\alpha$', fontsize='medium')
            plt.legend(fontsize='medium',loc='best')
            plt.tight_layout()
            
        
    def var_reconstruction_target(self, Lxy_Var = False, zr_Var = True,
                                  Lxy_var = 0.01, zr_var = 0.005, nxy_var = 2, nzr_var = 6):
        
        self.field_var = self.field
        if Lxy_Var != False:
            Zs_var = []; Alpha_var = []
            for n in range(nxy_var):
                self.field_var.zs(Lx = ((n+1)*Lxy_var+Lxy_var), Ly = ((n+1)*Lxy_var+Lxy_var),
                                  n_x = round((210*Lxy_var)*(n+1)), n_y = round((210*Lxy_var)*(n+1)),
                                  zr=0.0)
                Zs_var.append(self.field_var.Zs); Alpha_var.append(self.field_var.alpha[0,:])
            fig, axs = plt.subplots(nrows=2, ncols=1, dpi=200)
            fig.suptitle('Reconstruction area variation - Impedance', fontsize = 'medium')
            for n in range(nxy_var):
                axs[0].semilogx(self.freq_decomp, np.real(Zs_var[n]), label=f'Lxy = {Lxy_var*(n+1)}')
                axs[1].semilogx(self.freq_decomp, np.imag(Zs_var[n]), label=f'Lxy = {Lxy_var*(n+1)}')
            axs[0].set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs[1].set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs[1].set_ylim((-0.02, 1.02))
            axs[0].set_xlabel('Frequency [Hz]', fontsize = 'medium'); axs[1].set_xlabel('Frequency [Hz]', fontsize = 'medium')
            axs[0].set_ylabel('Frequency [Hz]', fontsize = 'medium'); axs[1].set_xlabel('Frequency [Hz]', fontsize = 'medium')
            axs[0].grid(color='gray', linestyle='-.', linewidth=1.5)
            axs[1].grid(color='gray', linestyle='-.', linewidth=0.4)
            
            fig, axs = plt.subplots(nrows=1, ncols=1, dpi=200)
            fig.suptitle('Reconstr. area analysis - Absorption', fontsize = 'large')
            for n in range(nxy_var):
                axs.semilogx(self.freq_decomp, np.array(Alpha_var[n]), label=f'Lxy = {Lxy_var*(n+1)}')
            axs.set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs.set_ylim((-0.02, 1.02))
            axs.set_xlabel('Frequency [Hz]', fontsize = 'medium'); 
            axs.set_ylabel('Abs. coefficient [-]', fontsize = 'medium'); 
            axs.grid(color='gray', linestyle='-.', linewidth=1.5)
            plt.legend(fontsize='medium', loc='best')
            plt.show()
            
        if zr_Var != False:
            Zs_var = []; Alpha_var = []
            for n in range(nzr_var):
                self.field_var.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, zr=(n+1)*zr_var+zr_var)
                Zs_var.append(self.field_var.Zs); Alpha_var.append(self.field_var.alpha[0,:])
                self.field_var.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, zr=((n+1)*zr_var+zr_var)*-1)
                Zs_var.append(self.field_var.Zs); Alpha_var.append(self.field_var.alpha[0,:])
            fig, axs = plt.subplots(nrows=2, ncols=1, dpi=200)
            fig.suptitle('Reconstr. area analysis - Impedance', fontsize = 'medium')
            for n in np.array(np.linspace(0,int(2*nxy_var),int(nxy_var),endpoint=False)):
                axs[0].semilogx(self.freq_decomp, np.real(Zs_var[int(n)]),
                                linewidth=2.0, label=f'$z_r$ = {(n+1)*zr_var+zr_var}')
                axs[0].semilogx(self.freq_decomp, np.real(Zs_var[int(n+1)]),
                                linewidth=2.0, label=f'$z_r$ = {((n+1)*zr_var+zr_var)*-1}')  
                axs[1].semilogx(self.freq_decomp, np.imag(Zs_var[int(n)]),
                                linewidth = 2.0, label=f'$z_r$ = {(n+1)*zr_var+zr_var}')
                axs[1].semilogx(self.freq_decomp, np.imag(Zs_var[int(n+1)]),
                                linewidth=2.0, label=f'$z_r$ = {((n+1)*zr_var+zr_var)*-1}')
            axs[0].set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs[1].set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs[0].set_xlabel('Frequency [Hz]', fontsize = 'medium'); axs[1].set_xlabel('Frequency [Hz]', fontsize = 'medium')
            axs[0].set_ylabel('Re{Z}', fontsize = 'medium'); axs[1].set_ylabel('Im{Z}', fontsize = 'medium')
            axs[0].grid(color='gray', linestyle='-.', linewidth=1.5)
            axs[1].grid(color='gray', linestyle='-.', linewidth=0.4)
            plt.legend(fontsize='medium', loc='best')
            plt.show()
            
            fig, axs = plt.subplots(nrows=1, ncols=1)
            fig.suptitle('Reconstr. area analysis - Absorption', fontsize = 'large')
            for n in np.array(np.linspace(0,int(2*nxy_var),int(nxy_var),endpoint=False)):
                axs.semilogx(self.freq_decomp, np.array(Alpha_var[int(n)]), linewidth=2.0, label=f'$z_r$ = {(n+1)*zr_var+zr_var}')
                axs.semilogx(self.freq_decomp, np.array(Alpha_var[int(n+1)]), linewidth=2.0, label=f'$z_r$ = {((n+1)*zr_var+zr_var)*-1}')
            axs.set_xlim((0.8*self.freq_decomp[0], 1.05*self.freq_decomp[-1]))
            axs.set_ylim((-0.02, 1.02))
            axs.set_xlabel('Frequency [Hz]', fontsize = 'medium'); 
            axs.set_ylabel('Abs. coefficient [-]', fontsize = 'medium'); 
            axs.grid(color='gray', linestyle='-.', linewidth=1.5)
            plt.legend(fontsize='medium', loc='best')
            plt.show()
            
    def plotIR(self, IR_lst, recIdx=0, place=None):
        xticks = [0, 0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009, 0.0105, 0.012, 0.0135, 0.015]
        xticksLabel = ['0,0', '1,5', '3,0', '4,5', '6,0', '7,5', '9,0', '10,5', '12,0', '13,5', '15,0']
        
        def _euclid_dist(r1,r2):
            d = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2])**2)
            return d
        dr = np.zeros((len(IR_lst),), dtype='float64')
        
        IRs_objs = IR_lst
        import numpy
        if type(IR_lst[0]) == numpy.ndarray:
            IRs_objs = []
            for i in range(len(IR_lst)):
                IR = pytta.classes.SignalObj(signalArray = IR_lst[i], 
                domain='time', samplingRate = self.fs, numSamples=len(IR_lst[i]))
                IR.comment = self.receivers.coord[i]
                IRs_objs.append(IR)
        else:
            pass
    
        for i in range(len(IRs_objs)):
            if i == 0:
                dr[i] = None
            else:
                dr[i] = _euclid_dist(IRs_objs[0].comment, IRs_objs[i].comment)
        idx_far = np.nanargmax(dr)
        
        if place == 'farthest':
            plt.figure(dpi=250)
            plt.plot(IRs_objs[idx_far].timeVector,IRs_objs[idx_far].timeSignal, linewidth=1.0)
            plt.title('Resp. Imp.: Ponto mais distante do centro')
            plt.xlim((0, 0.0155))
            plt.xticks(xticks, xticksLabel)
            plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
            plt.grid(linestyle = '--', which='both')
            plt.tight_layout()

        elif place == 'farthest pair':
            d_far = np.zeros((len(IRs_objs),), dtype='float64')
            for i in range(len(IRs_objs)):
                if i != idx_far:
                    d_far[i] = _euclid_dist(IRs_objs[idx_far].comment, IRs_objs[i].comment)
                else:
                        d_far[i]=None
                idx_far2 = np.nanargmin(d_far)
        
            if IRs_objs[idx_far].comment[2] > IRs_objs[idx_far2].comment[2]:
                IRup_pt = IRs_objs[idx_far]; IRdown_pt = IRs_objs[idx_far2]
            else:
                IRup_pt = IRs_objs[idx_far2]; IRdown_pt = IRs_objs[idx_far]
            plt.figure(dpi=250)
            plt.plot(IRup_pt.timeVector,IRup_pt.timeSignal, linewidth=1.0, label='Upper pt.')
            plt.plot(IRdown_pt.timeVector,IRdown_pt.timeSignal, linewidth=1.0, label='Lower pt.')
            plt.title('Resp. Imp.: Par de pontos mais distantes do centro')
            plt.xlim((0, 0.0155))
            plt.xticks(xticks, xticksLabel)
            plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
            plt.grid(linestyle = '--', which='both')
            plt.legend()
            plt.tight_layout()        
        else:
            cordX = round(IRs_objs[recIdx].comment[0],3)
            cordY = round(IRs_objs[recIdx].comment[1],3)
            cordZ = round(IRs_objs[recIdx].comment[2],3)
            plt.figure(dpi=250)
            plt.plot(IRs_objs[recIdx].timeVector,IRs_objs[recIdx].timeSignal, linewidth=1.0, label='Upper pt.')
            plt.title(f'Resp. Imp do ponto [{cordX}, {cordY}, {cordZ}]')
            plt.xlim((0, 0.0155))
            plt.xticks(xticks, xticksLabel)
            plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
            plt.grid(linestyle = '--', which='both')
            plt.tight_layout()       
                  
            
            
            
        
        # SSR.plot_alpha(alphas, freqs, save=True, leg=['rec.',r'from $P(k)$'],
        #                    name='alphas'+namee, path=path_res + '/' + namee + '/')
        
        
        
        
                    
        
        
    
        
        
        
        
        
        
        
       
        
        
        
