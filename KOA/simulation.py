
import numpy as np
from scipy.signal import butter,filtfilt


def dampened_step(zeta,harmonic,t):
    W=harmonic*2*np.pi
    phase=np.arccos(zeta)
    return 1-np.exp(-zeta*W*t)*np.sin(np.sqrt(1-zeta**2)*W*t+phase)/np.sin(phase)
def dampened_impulse(zeta,harmonic,t):
    W=2 * np.pi * harmonic
    return np.exp(-zeta * W * t) * np.sin(np.sqrt(1 - zeta**2) * W * t)

def complex_impulse(time_step,total_time,magnitude,frequency_data, dtype=np.float32):
    t= np.arange(0, total_time, time_step, dtype=dtype)
    y=np.zeros(t.shape, dtype=dtype)
    frequency_data[0,:]=frequency_data[0,:]/np.sum(frequency_data[0,:])
    for i in range(frequency_data.shape[1]):
        y+=dampened_impulse(frequency_data[2,i],frequency_data[1,i],t)*frequency_data[0,i]
    y*=magnitude
    return y

def complex_step(time_step,total_time,magnitude,frequency_data, dtype=np.float32):#data is a numpy array with amplitudes frequency and zeta variables
    t= np.arange(0, total_time, time_step,dtype=dtype)
    y=np.zeros(t.shape, dtype=dtype)
    frequency_data[0,:]=frequency_data[0,:]/np.sum(frequency_data[0,:])
    for i in range(frequency_data.shape[1]):
        y+=dampened_step(frequency_data[2,i],frequency_data[1,i],t)*frequency_data[0,i]
    y*=magnitude
    return y


class zv_shaper:
    def __init__(self,harmonic_frequency,zeta):
        self.K=0
        self.shifted_time=0
        self.set_parameters(harmonic_frequency,zeta)
        self.A=[]
        self.T=[]
        self.min_freq=21
    def get_parameters(self):
        return self.K,self.shifted_time 
    def set_parameters(self,harmonic_frequency,zeta):

        if harmonic_frequency==0:
            self.shifted_time=0
        else:
            df = np.sqrt(1. - zeta**2)
            self.shifted_time =   1/(harmonic_frequency*df)
            self.T=np.array([0,0.5*self.shifted_time])
            self.K= np.exp(-zeta*np.pi/df)
            A=np.array([1,self.K])
            self.A=A/np.sum(A)
            if np.isnan(self.shifted_time) or np.isnan(self.K):
                breakpoint()
    def get_impulses(self):
        return self.A,self.T
    def accelerate(self,time_step,total_time,acceleration,frequency_data):
        if self.shifted_time==0:
            if np.isnan(total_time):
                breakpoint()
            return complex_step(time_step,total_time,acceleration,frequency_data)
        else:
            a=complex_step(time_step,total_time,acceleration*self.A[0],frequency_data)
            total_len=len(a)
            for i in range(1,len(self.A)):
                a2=complex_step(time_step,total_time-self.T[i],acceleration*self.A[i],frequency_data)
                diff=total_len-len(a2)
                a2=np.pad(a2,(diff,0))
                a=np.add(a,a2)
            return a

class zvd_shaper:
    def __init__(self,harmonic_frequency,zeta):
        self.K=0
        self.shifted_time=0
        self.A=[]
        self.T=[]
        self.set_parameters(harmonic_frequency,zeta)
        self.min_freq=29

    def get_parameters(self):
        return self.K,self.shifted_time 

    def get_impulses(self):
        return self.A,self.T
    def set_parameters(self,harmonic_frequency,zeta):

        if harmonic_frequency==0:
            self.shifted_time=0
        else:
            df = np.sqrt(1. - zeta**2)
            self.shifted_time =  1/(harmonic_frequency*df)
            self.T=np.array([0,0.5*self.shifted_time,1*self.shifted_time])
            self.K= np.exp(-zeta*np.pi/df)
            A=np.array([1,2.*self.K,self.K**2])
            self.A=A/np.sum(A)
            if np.isnan(self.shifted_time) or np.isnan(self.K) or np.any(np.isnan(self.A)) or np.any(np.isnan(self.T)):
                breakpoint()
        
    def accelerate(self,time_step,total_time,acceleration,frequency_data,mode='real'):
        if self.shifted_time==0:
            if np.isnan(total_time):
                breakpoint()
            return complex_step(time_step,total_time,acceleration,frequency_data)
        else:
            a=complex_step(time_step,total_time,acceleration*self.A[0],frequency_data)
            total_len=len(a)
            for i in range(1,len(self.A)):
                a2=complex_step(time_step,total_time-self.T[i],acceleration*self.A[i],frequency_data)
                diff=total_len-len(a2)
                a2=np.pad(a2,(diff,0))
                a=np.add(a,a2)
            return a


class mzv_shaper:
    def __init__(self,harmonic_frequency,zeta):
        self.K=0
        self.shifted_time=0
        self.A=[]
        self.T=[]
        self.set_parameters(harmonic_frequency,zeta)
        self.min_freq=23

    def get_parameters(self):
        return self.K,self.shifted_time 
    def get_impulses(self):
        return self.A,self.T
    def set_parameters(self,harmonic_frequency,zeta):
        if harmonic_frequency==0:
            self.shifted_time=0
            return
        else:
            df = np.sqrt(1. - zeta**2)
            self.shifted_time =  1/(harmonic_frequency*df)
            self.T=np.array([0,0.375*self.shifted_time,.75*self.shifted_time])
            self.K= np.exp(-0.75*zeta*np.pi/df)
            a1 = 1. - 1. / np.sqrt(2.)
            A=np.array([a1,(np.sqrt(2.) - 1.) * self.K ,a1 * self.K**2])
            self.A=A/np.sum(A)
            if np.isnan(self.shifted_time) or np.isnan(self.K) or np.any(np.isnan(self.A)) or np.any(np.isnan(self.T)):
                breakpoint()
        
    def accelerate(self,time_step,total_time,acceleration,frequency_data):
        if self.shifted_time==0:
            if np.isnan(total_time):
                breakpoint()
            return complex_step(time_step,total_time,acceleration,frequency_data)
        else:
            a=complex_step(time_step,total_time,acceleration*self.A[0],frequency_data)
            total_len=len(a)
            for i in range(1,len(self.A)):
                a2=complex_step(time_step,total_time-self.T[i],acceleration*self.A[i],frequency_data)
                diff=total_len-len(a2)
                a2=np.pad(a2,(diff,0))
                a=np.add(a,a2)
            return a

class modified_shaper:
    def __init__(self):
        self.A=np.array([1])
        self.T=np.array([0])
        self.shifted_time=0
    def set_impulses(self,A,T):
        self.shifted_time=T[-1]
        self.T=np.concatenate((np.zeros(1),T),axis=0)
        self.A=A/A.sum()
    def get_impulses(self):
        return self.A,self.T
    def accelerate(self,time_step,total_time,acceleration,frequency_data):
        if self.shifted_time==0:
            if np.isnan(total_time):
                breakpoint()
            return complex_step(time_step,total_time,acceleration,frequency_data)
        else:
            a=complex_step(time_step,total_time,acceleration*self.A[0],frequency_data)
            total_len=len(a)
            for i in range(1,len(self.A)):
                a2=complex_step(time_step,total_time-self.T[i],acceleration*self.A[i],frequency_data)
                diff=total_len-len(a2)
                a2=np.pad(a2,(diff,0))
                a=np.add(a,a2)
            return a



def PRV(A,T,frequency_response,max_components=0,mode='median'):
    amplitudes=0
    frequencies=0
    zetas=0

    if max_components!=0:
        indicies=np.argpartition(frequency_response[0,:],-max_components)[-max_components:]
        amplitudes=frequency_response[0,indicies]
        amplitudes=amplitudes/np.sum(amplitudes)
        frequencies=2*np.pi*frequency_response[1,indicies]
        zetas=frequency_response[2,indicies]
    else:
        frequency_data=frequency_response.copy()
        amplitudes=frequency_data[0,:]
        frequencies=2*np.pi*frequency_data[1,:]
        zetas=frequency_data[2,:]
        
    inv_D=1/A.sum()
    dampened_frequencies=frequencies*np.sqrt(1-np.square(zetas))
    V =   np.exp(-zetas*frequencies* T[-1])
    W = A * np.exp(np.outer(zetas*frequencies, T))
    S = W * np.sin(np.outer(dampened_frequencies, T))
    C = W * np.cos(np.outer(dampened_frequencies, T))
    percentage_residual_vibration=V*np.sqrt(S.sum(axis=1)**2 + C.sum(axis=1)**2) * inv_D
    if mode=='median':
        value= np.average(percentage_residual_vibration,weights=amplitudes)*100
        return np.clip(value,a_min=0,a_max=100)
    if mode=='max':
        max_freq_index=amplitudes.argmax()
        value=percentage_residual_vibration[max_freq_index]*100
        return np.clip(value,a_min=0,a_max=100)
    if mode=='vals':
        return percentage_residual_vibration
    

def ERVA(shaper, frequency_data):
    A,T=shaper.get_impulses()
    vals = PRV(A,T, frequency_data, max_components=0,mode='vals')
    # The input shaper can only reduce the amplitude of vibrations by
    # SHAPER_VIBRATION_REDUCTION times, so all vibrations below that
    # threshold can be igonred
    psd=frequency_data[0,:]
    vibr_threshold = psd.max() / 20
    remaining_vibrations = np.maximum(
            vals * psd - vibr_threshold, 0).sum()
    all_vibrations = np.maximum(psd - vibr_threshold, 0).sum()
    return remaining_vibrations / all_vibrations
    
class ideal_shaper:
    def accelerate(time_step,total_time,acceleration,frequency_data):
        t= np.arange(0, total_time, time_step)
        y= np.full(t.shape,acceleration)
        return y

def shaped_ideal_signal(shaper,time_step,total_time):
    t= np.arange(0, total_time, time_step)
    a=np.full(t.shape,0)
    total_len=len(a)
    A,T=shaper.get_impulses()
    for i in range(1,len(self.A)):
        t_2= np.arange(0, total_time-T[i], time_step)
        a2=np.full(t2.shape,A[i])
        diff=total_len-len(a2)
        a2=np.pad(a2,(diff,0))
        a=np.add(a,a2)
    return a

class toolhead:
    def __init__(self,max_accel,max_speed,time_step,shaper,frequency_response):
        self.accel=max_accel
        self.max_speed=max_speed
        self.shaper=shaper
        self.time_step=time_step
        self.frequency_response=frequency_response
        self.noise=True
        self.noise_accelerometer_std=100
        self.noise_random=0.05
        
    def trapezoidal_generator(self,distance):
        if distance<0:
            direction=-1
        else:
            direction=1
        distance=np.abs(distance)
        v_max=np.sqrt(distance/(3*self.accel))*self.accel
        if v_max>self.max_speed:
            v_max=self.max_speed
        accel_time=v_max/self.accel
        accel_dist=0.5*self.accel*accel_time**2
        total_time = 2 * accel_time + (distance - 2 * accel_dist) / v_max
        return total_time,accel_time,total_time-accel_time,direction
        
    def add_accelerations(self,a1,a2):
        total_len=len(a1)
        diff=total_len-len(a2)
        a2=np.pad(a2,(diff,0))
        return np.add(a1,a2)
        
    def move(self,positions,extra_sim_time=0.3,dtype=np.float32):
        moves = np.zeros(len(positions) - 1, dtype=[
            ('accel_time', float),
            ('decel_time', float),
            ('move_start', float),
            ('direction',int),
            ('move_end',float),

        ])
        move_start = 0
        acceleration=self.accel
        for i in range(1, len(positions)):
            distance = positions[i] - positions[i - 1]
            total_time, accel_time, decel_time,direction = self.trapezoidal_generator(distance)
            move_end = move_start + total_time
            moves[i - 1] = (accel_time, decel_time, move_start,direction,move_end)
            move_start = move_end
        total_time=move_start+extra_sim_time
        t= np.arange(0, total_time, self.time_step)
        a=np.zeros(t.shape,dtype)
        if np.isnan(total_time):
            breakpoint()
        for i in range(0,len(positions)-1):
            if self.noise==True:
                a=self.add_accelerations(a,complex_impulse(self.time_step,total_time-moves[i]['move_start'],moves[i]['direction']*acceleration,self.frequency_response))   
            
            a=self.add_accelerations(a,self.shaper.accelerate(self.time_step,total_time-moves[i]['move_start'],moves[i]['direction']*acceleration,self.frequency_response))
            a=self.add_accelerations(a,self.shaper.accelerate(self.time_step,total_time-moves[i]['move_start']-moves[i]['accel_time'],-moves[i]['direction']*acceleration,self.frequency_response))
            
            a=self.add_accelerations(a,self.shaper.accelerate(self.time_step,total_time-moves[i]['move_start']-moves[i]['decel_time'],-moves[i]['direction']*acceleration,self.frequency_response))
            if moves[i]['move_end']!=total_time:
               a=self.add_accelerations(a,self.shaper.accelerate(self.time_step,total_time-moves[i]['move_end'],moves[i]['direction']*acceleration,self.frequency_response))
        if self.noise:
            gaussian_noise = np.random.normal(0, acceleration*self.noise_random, a.shape)
            accelerometer_noise= np.random.normal(0, self.noise_accelerometer_std, a.shape)
            a=a+accelerometer_noise+gaussian_noise
            #filtering
            fs=1/self.time_step
            cutoff = 150      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 4       # sin wave can be approx represented as quadratic
            n = int(total_time * fs) # total number of samples
            def butter_lowpass_filter(data, cutoff, fs, order):
                normal_cutoff = cutoff / nyq
                # Get the filter coefficients 
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y
            a = butter_lowpass_filter(a, cutoff, fs, order)


        return t,a
        