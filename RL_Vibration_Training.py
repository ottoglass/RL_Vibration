
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from copy import copy
import torch

from glob import glob
#---Logger---#
from wandb.integration.sb3 import WandbCallback
import wandb
from motan.shaper_calibrate_simplified import freq_from_raw_data
from typing import Callable

from KOA.simulation import mzv_shaper,zv_shaper,zvd_shaper
from KOA.environment import PrinterV0,PrinterV2
from KOA.FeatureExtractor import RNN_Extractor,RNN_Model,RNN_ExtractorV2,T5_Extractor
from typing import Callable

class ModelTrainingManipulation(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
    parent: EvalCallback

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True
        # self.model.learning_rate=0.001

        
        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training


class DynamicLearningRateCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super(DynamicLearningRateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check if it's time to update the learning rate
        if self.n_calls % self.check_freq == 0:
            # Retrieve the current reward
            if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
                mean_reward = np.mean([info['episode']['r'] for info in self.locals['infos'] if 'episode' in info])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f} - Best mean reward: {self.best_mean_reward:.2f}")

                # Adjust learning rate based on mean reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Increase learning rate if reward improves
                    new_lr = self.model.learning_rate * 1.1
                else:
                    # Decrease learning rate if reward doesn't improve
                    new_lr = self.model.learning_rate * 0.9

                # Ensure learning rate doesn't become too small or too large
                new_lr = np.clip(new_lr, 1e-5, 1e-3)
                self.model.learning_rate = new_lr

                # Apply the new learning rate to the optimizers
                for optimizer in self.model.policy.optimizer.param_groups:
                    optimizer['lr'] = new_lr

                if self.verbose > 0:
                    print(f"New learning rate: {new_lr:.6f}")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

if __name__ == '__main__':
    STEPS_PER_EPISODE=50
    torch.autograd.set_detect_anomaly(True)
   # Example usage
    max_velocity = 1000 # mm/s
    acceleration = 10000 # mm/s^2
    time_step = 0.001  # s
    harmonic_frequency=50
    zeta=0.1
    num_actions=2

    use_wandb=True
    df = pd.read_csv('ReinforcementLearningCourse-main\\resonances_x_20240420_042935.csv')
    df_largest=df.nlargest(200, 'psd_x')
    frequencies=df_largest["freq"].to_numpy()
    amplitudes=df_largest["psd_x"].to_numpy()
    # amplitudes=df_largest["psd_xyz"].to_numpy()
    zetas=np.random.uniform(low=0.0,high=.4,size=frequencies.size)
    # frequency_response=[]
    frequency_response=np.vstack((amplitudes,frequencies,zetas))
    df_y = pd.read_csv('ReinforcementLearningCourse-main\\resonances_y_20240528_093025.csv')
    frequencies=df_y["freq"].to_numpy()
    amplitudes=df_y["psd_y"].to_numpy()
    zetas=np.random.uniform(low=0.0,high=.4,size=frequencies.size)
    frequency_response_y=np.vstack((amplitudes,frequencies,zetas))

    shaper=zvd_shaper(harmonic_frequency,zeta)
    klipper_dataset=[freq_from_raw_data(file) for file in glob("ReinforcementLearningCourse-main\\klipper_thread_dataset\**\\raw*.csv")]

    moves=[50,150,50]#50,150,50,150]
    # moves=[50,150,50]
    def make_env():
        return PrinterV2(max_velocity,acceleration,time_step,copy(shaper),klipper_dataset,moves,STEPS_PER_EPISODE=STEPS_PER_EPISODE,num_actions=num_actions)

    # env= [make_env for i in range(4)]
    env=Monitor(make_env())
    eval_env=PrinterV2(max_velocity,acceleration,time_step,copy(shaper),[frequency_response,frequency_response_y],moves,STEPS_PER_EPISODE=STEPS_PER_EPISODE,num_actions=num_actions)
    eval_env.eval()
    eval_env.artificial_probability=0
    eval_env=Monitor(eval_env)
    # env = Monitor(DummyVecEnv(env))
    config={}
    policy_kwargs = dict(
        features_extractor_class=T5_Extractor,
        features_extractor_kwargs=dict(features_dim=128),#weights='pre_trained_weights.pth'),
    )
    n_action_noise=NormalActionNoise(mean=np.zeros(shape=(env.unwrapped.num_actions,)),sigma=np.ones(shape=(env.unwrapped.num_actions,))*0.1
    
    )
    

    if use_wandb:
        config = {
            "policy_type": "T5-Policy",
            "total_timesteps": 1000000,
            "env_name": "Printer-V2",
        }
        run = wandb.init(
            project="Vibration",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional-
        )
        # Stop training if there is no improvement after more than 3 evaluations
        # stop_train_callback = ModelTrainingManipulation(max_no_improvement_evals=10, min_evals=10, verbose=1)
        eval_callback = EvalCallback(eval_env, eval_freq=STEPS_PER_EPISODE*200,n_eval_episodes=1,verbose=1,best_model_save_path=f'models/{run.id}/best_model')#, callback_after_eval=stop_train_callback, )
        #MODEL
        # model=SAC.load("ReinforcementLearningCourse-main\\261-best-TD3-ZVD.zip",env=env)
        # model=TD3.load("models\\eq3cmwti\\best_model\\best_model",env=env,device='cpu')
        # model=TD3.load("models\\dbid2m52\\best_model\\best_model",env=env,device='cpu')
        # model=TD3.load("models\\gwwniiga\\best_model\\best_model",env=env,device='cpu')
        model=TD3.load("models\\ruhcnz86\\best_model\\best_model",env=env,device='cpu')
        model.learning_rate=linear_schedule(1e-4)

        # model= TD3("MlpPolicy",env,verbose=1,tensorboard_log=f"runs/{run.id}",buffer_size=int(7e5),device="cpu",policy_kwargs=policy_kwargs,action_noise=n_action_noise,learning_rate=linear_schedule(1e-3),train_freq=(50,"episode"))

        model.learn(total_timesteps=700000,log_interval=STEPS_PER_EPISODE,callback=
                [
                    WandbCallback(verbose=2),#gradient_save_freq=100,model_save_path=f"models/{run.id}",model_save_freq=1000
                    #DynamicLearningRateCallback(check_freq=STEPS_PER_EPISODE, verbose=1),
                    eval_callback,
                ]
            )
    else:
        # stop_train_callback = ModelTrainingManipulation(max_no_improvement_evals=10, min_evals=5, verbose=1)
        eval_callback = EvalCallback(eval_env, eval_freq=STEPS_PER_EPISODE*500,verbose=1,n_eval_episodes=5)#, callback_after_eval=stop_train_callback, verbose=1)
        model= SAC("MlpPolicy",env,verbose=1,tensorboard_log=f"runs/0",buffer_size=300000,device="cpu",policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=100000,log_interval=10,callback=eval_callback)


    obs = env.reset()
    action, _states = model.predict(obs[0])
    print(action)

    if use_wandb:
        run.finish()