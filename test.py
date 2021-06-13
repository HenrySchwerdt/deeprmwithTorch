import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import environment
import parameters
import pg_network
import numpy as np
import torch as T
import util
import other_agents

pa = parameters.Parameters()
pa.compute_dependent_parameters()
env = environment.Env(pa, render=True)

agent = pg_network.Agent(pa)

#loads best computed models
agent.load_models()

observation = env.observe()
done = False

total_reward = 0
# deeprm network
while not done:
    action = agent.q_eval.forward(T.tensor(observation)).argmax()
    print(action)
    observation, reward, done, info = env.step(action.item())
    total_reward += reward

env.reset()
print(total_reward)
total_reward = 0
done = False
#get_packer_action
print('packer')
while not done:
    action = other_agents.get_packer_action(env.machine,env.job_slot)
    observation_, reward, done, info = env.step(action)
    total_reward += reward

print(total_reward)

