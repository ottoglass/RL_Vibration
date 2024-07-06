
import numpy as np
from scipy.signal import spectrogram
import gymnasium as gym
from gymnasium import spaces
from KOA.simulation import toolhead,ideal_shaper,PRV,ERVA,modified_shaper
from KOA.shaper_calibrate import fit_shaper
from KOA.shaper_calibrate import SSR

def fix_nan(arr):
    ok = ~np.isnan(arr)
    xp = ok.ravel().nonzero()[0]
    fp = arr[~np.isnan(arr)]
    x  = np.isnan(arr).ravel().nonzero()[0]
    # Replacing nan values
    if np.isnan(arr).all():
        return False
    arr[np.isnan(arr)] = np.interp(x, xp, fp)
    return arr



class PrinterV0(gym.Env):
    def __init__(self,max_velocity,acceleration,time_step,shaper,frequency_response,moves,render_mode='human',STEPS_PER_EPISODE=50,lr_scheduler=None,num_actions=2):
        super(PrinterV0, self).__init__()

        # Define action and observation spaces
        # self.action_space = spaces.Dict({
        #     'frequency': spaces.Box(low=0,high=128,shape=(1,),dtype=np.float32),  # 0-128 Hz
        #     'zeta': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # 0-1
        # })
        if len(frequency_response)==0:
            raise ValueError("Missing frequecy data!")
        self.initial_frequency_response=frequency_response.copy()
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
        self.printer=toolhead(acceleration,max_velocity,time_step,shaper,self.get_random_data())
        ideal_printer=toolhead(acceleration,max_velocity,time_step,ideal_shaper,self.get_random_data())
        ideal_printer.noise=False
        self.iter_counter=0
        self.ideal_curve=ideal_printer.move(moves)[1]
        time_start=0.3/self.printer.time_step
        self.lr=lr_scheduler
        obs=self.preprocessing(self.ideal_curve[-int(time_start)-1:-1])
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs.shape, dtype=np.float32)  # Example observation shape
        self.moves=moves
        self.render_mode=render_mode
        self.artificial_probability=0.9
        self.artificial_probability_decay=0.005
        self.episode_count=0
        self.best_reward=-9999999999
        self.alphas=0
        self.noise_probability=0.9
        self.eval_mode=False
        self.state=np.zeros(2)
        self.ideal_state=np.zeros(2)
        self.vibration_threshold=5
        self.STEPS_PER_EPISODE=STEPS_PER_EPISODE
        self.num_actions=num_actions
        self.max_ratio=0.4

    def eval(self,full=False):
        self.artificial_probability=0
        self.eval_mode=True
        if full:
            self.noise_probability=0


    def get_random_data(self):
        random_index=np.random.randint(low=0,high=len(self.initial_frequency_response))
        return self.initial_frequency_response[random_index]
        
    def reset(self,seed=None,options=None):
        # Reset the environment
        # Return initial observation
        super().reset(seed=seed)
        self.iter_counter=0
        np.random.seed=(seed)
        # freq,zeta=np.random.rand(2)
        # freq,zeta=(0,0)
        # self.printer.shaper.set_parameters(freq*150,zeta*0.99)
        self.printer.noise=np.random.uniform()<self.noise_probability
        artificial=np.random.uniform()<np.exp(self.artificial_probability_decay*self.episode_count)*self.artificial_probability
        if not self.eval_mode:
            if artificial:
                frequency_response=np.random.uniform(low=0,high=1,size=(3,2))
                frequency_response[1,:]*=128+5
                frequency_response[2,:]=np.ones(frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=0.3)
                # frequency_response[2,:]=np.random.uniform(low=0.01,high=0.8,size=frequency_response[2,:].shape)
                self.printer.frequency_response=frequency_response
            else:
                initial_frequency_response=self.get_random_data()
                variation=np.random.uniform(low=0.9,high=1.1,size=initial_frequency_response.shape).reshape(initial_frequency_response.shape)
                frequency_response=np.multiply(initial_frequency_response,variation)
                # frequency_response[2,:]=np.random.uniform(low=0.01,high=0.4,size=variation[2,:].shape)
                frequency_response[2,:]=np.ones(frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=0.3)

                self.printer.frequency_response=frequency_response
        else:
            initial_frequency_response=self.get_random_data()
            initial_frequency_response[2,:]=np.ones(initial_frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=self.max_ratio)

            # initial_frequency_response[2,:]=np.random.uniform(low=0.01,high=0.2,size=initial_frequency_response[2,:].shape)
            self.printer.frequency_response=initial_frequency_response

        res=fit_shaper(self.printer.shaper,self.printer.frequency_response)
        max_frequency=res.freq
        # max_freq_index=self.printer.frequency_response[0,:].argmax()
        initial_zeta=0.1
        # max_frequency=self.printer.frequency_response[1,max_freq_index]
        #estimate best input shaper values
        best_ratio=self.printer.frequency_response[2,0]
        test_frequencies=np.arange(self.printer.shaper.min_freq,150,0.1)
        test_results=np.zeros(test_frequencies.shape)
        # test_ratios=np.arange(0,0.4,0.002)
        # test_results=np.zeros((len(test_frequencies),len(test_ratios)))
        # for i,frequency in enumerate(test_frequencies):
        #     for j,ratio in enumerate(test_ratios):
        #         self.printer.shaper.set_parameters(frequency,ratio)
        #         A,T=self.printer.shaper.get_impulses()
        #         test_results[i]=PRV(A,T,self.printer.frequency_response)
        for i,frequency in enumerate(test_frequencies):
            self.printer.shaper.set_parameters(frequency,best_ratio)
            A,T=self.printer.shaper.get_impulses()
            test_results[i]=PRV(A,T,self.printer.frequency_response)
        best_frequency=test_frequencies[np.argmin(test_results)]
        #set initial input shaper values
        self.ideal_state[1]=best_ratio
        self.ideal_state[0]=best_frequency
        self.state=np.array([max_frequency,initial_zeta])
        self.printer.shaper.set_parameters(max_frequency,initial_zeta)
        return self._get_obs(),{}


    def step(self, action): 
        # Execute one time step within the environment
        # Update the environment state based on the action
        # Calculate the reward
        # Check if episode is done
        # Return observation, reward, done, info
        if np.any(np.isnan(action)):
            print("Received NaN action, clipping to zero")
            action = np.zeros_like(action)
        ideal_norm=0
        penalty=0
        if self.num_actions==1:
            self.state[1]=self.state[1]+action[0]*0.1
            # self.state[1]=action[0]*0.499+0.5
            self.state[1]=np.clip(self.state[0],0.001,max_ratio)
            ideal_norm=np.linalg.norm((self.ideal_state-self.state))/self.max_ratio
            penalty=ideal_norm
            # penalty=np.abs((self.ideal_state-self.state))/self.max_ratio
        else:
            self.state[0]=self.state[0]+action[0]
            self.state[1]=self.state[1]+action[1]*0.1
            self.state[1]=np.clip(self.state[0],0.001,self.max_ratio)
            self.state[0]=np.clip(self.state[0],self.printer.shaper.min_freq,150)
            ideal_norm=np.linalg.norm((self.ideal_state-self.state)/np.array([150,self.max_ratio/1.5]))
            penalty=ideal_norm*1.3
            # penalty=np.abs((self.ideal_state-self.state)/np.array([150,self.max_ratio])).sum()/2
        self.printer.shaper.set_parameters(self.state[0],self.state[1])#set parameters for the input shaper
        observation=self._get_obs()#generates the waveform
        #done
        self.iter_counter += 1
        truncated = self.iter_counter >= (self.STEPS_PER_EPISODE)            
        failed=False
        # terminated=False
        if np.any(np.isnan(observation)):
            observation=fix_nan(observation)
            if not observation:
                failed=True
        reward=self._calculate_reward(self.waveform)+1/((ideal_norm+0.1))
        # reward=1/((ideal_norm+0.1))
        if np.any(np.isnan(reward)):
            reward=-999999999
        if reward>=1/self.vibration_threshold:
            terminated=True
        terminated=False
        # penalty = np.sum(np.square(np.clip(action, -0.95, 0.95) - action))*100
        if truncated or terminated:
            self.episode_count+=1
        if self.eval_mode:
            # return observation,-SSR(self.printer.shaper,self.printer.frequency_response),truncated ,terminated, {}
            A,T=self.printer.shaper.get_impulses()
            return observation,-PRV(A,T,self.printer.frequency_response,mode='median',max_components=0),truncated ,terminated, {}
            # return observation,-ERVA(self.printer.shaper,self.printer.frequency_response)*100,truncated ,terminated, {}

        # if (terminated or truncated) and self.lr is not None:
        #     self.lr.add_episode(np.tanh((reward-penalty)*0.05))
        return observation,np.tanh((reward-penalty)/2),truncated ,terminated, {}
    
    def _get_waveform(self):
        
        self.waveform = self.printer.move(self.moves)[1]

    def _get_obs(self):
        # Generate moves and obtain waveform
        self._get_waveform()
        time_start=0.3/self.printer.time_step
        obs=self.preprocessing(self.waveform[-int(time_start)-1:-1])
        return obs


    def _calculate_reward(self, waveform):
            
        A,T=self.printer.shaper.get_impulses()
        # percentage_residual_vibration =PRV(A,T,self.printer.frequency_response,max_components=0,mode='median')*100
        percentage_residual_vibration=ERVA(self.printer.shaper,self.printer.frequency_response)*100
        # score=SSR(self.printer.shaper,self.printer.frequency_response)*100
        epsilon = 1e-9  # Small constant to prevent division by zero
        max_reward = 1000  # Maximum reward value to clip to
        smooth_reward = 1 / (percentage_residual_vibration + epsilon)#+score)
        smooth_reward = min(smooth_reward, max_reward)
        reward=smooth_reward
        return np.tanh((reward)*0.05)

    @staticmethod
    def preprocessing(signal,time_step= 0.001):
        # _,_,obs=spectrogram(signal)#,1/self.printer.time_step)
        obs=signal/np.max(np.abs(signal))
        v=np.cumsum(obs)*time_step
        v=v/np.max(np.abs(v))
        obs=np.expand_dims(np.transpose(np.vstack([v,obs])),axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed()



class PrinterV2(gym.Env):
    def __init__(self,max_velocity,acceleration,time_step,shaper,frequency_response,moves,render_mode='human',STEPS_PER_EPISODE=50,lr_scheduler=None,num_actions=2):
        super(PrinterV2, self).__init__()
        if len(frequency_response)==0:
            raise ValueError("Missing frequecy data!")
        self.initial_frequency_response=frequency_response.copy()
        self.num_actions=(num_actions*2)+1
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.printer=toolhead(acceleration,max_velocity,time_step,modified_shaper(),self.get_random_data())
        self.iter_counter=0
        time_start=0.3/self.printer.time_step
        self.lr=lr_scheduler
        ideal_printer=toolhead(acceleration,max_velocity,time_step,ideal_shaper,self.get_random_data())
        ideal_printer.noise=False
        self.iter_counter=0
        self.ideal_curve=ideal_printer.move(moves)[1]
        obs=self.preprocessing(self.ideal_curve[-int(time_start)-1:-1])
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs.shape, dtype=np.float32)  # Example observation shape
        self.moves=moves
        self.render_mode=render_mode
        self.artificial_probability=0.9
        self.artificial_probability_decay=0.0005
        self.episode_count=0
        self.best_reward=-9999999999
        self.alphas=0
        self.noise_probability=0.9
        self.eval_mode=False
        self.vibration_threshold=5
        self.STEPS_PER_EPISODE=STEPS_PER_EPISODE
        self.max_ratio=0.4

    def eval(self,full=False):
        self.artificial_probability=0
        self.eval_mode=True
        if full:
            self.noise_probability=0


    def get_random_data(self):
        random_index=np.random.randint(low=0,high=len(self.initial_frequency_response))
        return self.initial_frequency_response[random_index]
        
    def reset(self,seed=None,options=None):
        # Reset the environment
        # Return initial observation
        super().reset(seed=seed)
        self.iter_counter=0
        # np.random.seed=(seed)
        self.printer.noise=np.random.uniform()<self.noise_probability
        artificial=np.random.uniform()<np.exp(self.artificial_probability_decay*self.episode_count)*self.artificial_probability
        if not self.eval_mode:
            if artificial:
                frequency_response=np.random.uniform(low=0,high=1,size=(3,1))
                frequency_response[1,:]=frequency_response[1,:]*70+20
                frequency_response[2,:]=np.ones(frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=0.3)
                # frequency_response[2,:]=np.random.uniform(low=0.01,high=0.8,size=frequency_response[2,:].shape)
                self.printer.frequency_response=frequency_response
            else:
                initial_frequency_response=self.get_random_data()
                variation=np.random.uniform(low=0.9,high=1.1,size=initial_frequency_response.shape).reshape(initial_frequency_response.shape)
                frequency_response=np.multiply(initial_frequency_response,variation)
                # frequency_response[2,:]=np.random.uniform(low=0.01,high=0.4,size=variation[2,:].shape)
                frequency_response[2,:]=np.ones(frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=0.3)

                self.printer.frequency_response=frequency_response
        else:
            initial_frequency_response=self.get_random_data()
            initial_frequency_response[2,:]=np.ones(initial_frequency_response[2,:].shape)*np.random.uniform(low=0.01,high=self.max_ratio)

            # initial_frequency_response[2,:]=np.random.uniform(low=0.01,high=0.2,size=initial_frequency_response[2,:].shape)
            self.printer.frequency_response=initial_frequency_response

        return self._get_obs(),{}


    def step(self, action): 
        # Execute one time step within the environment
        # Update the environment state based on the action
        # Calculate the reward
        # Check if episode is done
        # Return observation, reward, done, info
        if np.any(np.isnan(action)):
            print("Received NaN action, clipping to zero")
            action = np.zeros_like(action)
        ideal_norm=0
        split=int(np.ceil(len(action)/2))
        A=np.array((action[0:split]*0.8)+0.1)
        T=np.array(action[split:]/20)
        self.printer.shaper.set_impulses(A,T)
        observation=self._get_obs()#generates the waveform
        #done
        self.iter_counter += 1
        truncated = self.iter_counter >= (self.STEPS_PER_EPISODE)

        reward=self._calculate_reward(self.waveform)
        # terminated=False
        failed=False
        # terminated=False
        if np.any(np.isnan(observation)):
            observation=fix_nan(observation)
            if type(observation) is bool:
                failed=True
                self.printer.shaper.set_impulses(np.array([0,.1]),np.array([0.99,0.01]))
                observation=self._get_obs()#generates the waveform

        if np.any(np.isnan(reward)) or failed:
            reward=-1
        # if reward>=1/self.vibration_threshold:
        #     terminated=True
        terminated=False
        if truncated or terminated:
            self.episode_count+=1
        if self.eval_mode:
            # return observation,-SSR(self.printer.shaper,self.printer.frequency_response),truncated ,terminated, {}
            A,T=self.printer.shaper.get_impulses()
            return observation,-PRV(A,T,self.printer.frequency_response,mode='median',max_components=0),truncated ,terminated, {}
            # return observation,-ERVA(self.printer.shaper,self.printer.frequency_response)*100,truncated ,terminated, {}

        # if (terminated or truncated) and self.lr is not None:
        #     self.lr.add_episode(np.tanh((reward-penalty)*0.05))
        return observation,reward,truncated ,terminated, {}
    
    def _get_waveform(self):
        
        self.waveform = self.printer.move(self.moves)[1]

    def _get_obs(self):
        # Generate moves and obtain waveform
        self._get_waveform()
        time_start=0.3/self.printer.time_step
        obs=self.preprocessing(self.waveform[-int(time_start)-1:-1])
        return obs


    def _calculate_reward(self, waveform):    
        A,T=self.printer.shaper.get_impulses()
        psd=self.printer.frequency_response[0,:]
        vals =PRV(A,T,self.printer.frequency_response,max_components=0,mode='vals')
        vibr_threshold = psd.max() / 20
        remaining_vibrations = np.maximum(
        vals * psd - vibr_threshold, 0).sum()
        all_vibrations = np.maximum(psd - vibr_threshold, 0).sum()
        normalized_klipper_vibrations=(remaining_vibrations / all_vibrations)*100
        epsilon = 1e-9  # Small constant to prevent division by zero
        max_reward = 1000  # Maximum reward value to clip to
        smooth_reward = 1 / (normalized_klipper_vibrations + epsilon)#+score)
        smooth_reward = min(smooth_reward, max_reward)
        reward=smooth_reward
        return np.tanh((reward)*0.05)-np.average(vals,weights=psd)*0.5

    @staticmethod
    def preprocessing(signal,time_step= 0.001):
        # _,_,obs=spectrogram(signal)#,1/self.printer.time_step)
        obs=signal/np.max(np.abs(signal))
        v=np.cumsum(obs)*time_step
        v=v/np.max(np.abs(v))
        obs=np.expand_dims(np.transpose(np.vstack([v,obs])),axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed()