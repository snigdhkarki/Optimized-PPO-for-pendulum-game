import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os



env_id = "Pendulum-v1"
n_envs = 4
total_timesteps = 1e5
# n_steps = 1024
# gae_lambda = 0.95
# gamma = 0.9
n_epochs = 10
ent_coef = 0.0
# learning_rate = 1e-3
# clip_range = 0.2
use_sde = True
sde_sample_freq = 4


LOG_DIR = './logs/'
OPT_DIR = './opt/'

def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 640, 1280),               #must be divisible by 64(batch size of PPO) 
        'gamma':trial.suggest_float('gamma', 0.8, 0.9999, log=True),
        'learning_rate':trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'clip_range':trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_float('gae_lambda', 0.8, 0.99)
    }
def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial) 
         
        env = make_vec_env(env_id,n_envs=n_envs,seed=0)
        # env = Monitor(env, LOG_DIR)
        # env = DummyVecEnv([lambda: env])        
        # env = VecFrameStack(env, 4, channels_order='last')      #most likely used when want 4 frames as input like for pong, channel_order not needed mostly default work i guess
       
        model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=0,n_epochs=n_epochs, ent_coef=ent_coef,use_sde=use_sde, sde_sample_freq=sde_sample_freq, **model_params)
        model.learn(total_timesteps=total_timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        print(e)
        return -1000

study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=20, n_jobs=1) #n_trails says how many models you want
print(study.best_params)
print(study.best_trial)







# env = make_vec_env(env_id,n_envs=n_envs,seed=0)
# model = PPO.load(os.path.join(OPT_DIR, 'trial_18_best_model.zip'))

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render("human")



