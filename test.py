import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import enivronment
import parameters
import pg_network
import numpy as np
import torch as T
import util
import other_agents

pa = parameters.Parameters()
env = enivronment.Env(pa, render=True)

agent = pg_network.Agent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=[pa.network_input_height,pa.network_input_width],
                     n_actions=pa.network_output_dim, mem_size=50000, eps_min=0.1,
                     batch_size=100, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='deep-rm')

#loads best computed models
agent.load_models()

observation = env.observe()
done = False


# deeprm network
while not done:
    action = agent.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    print(reward)

env.reset()
done = False
#get_packer_action
print('packer')
while not done:
    action = other_agents.get_packer_action(env.machine,env.job_slot)
    observation_, reward, done, info = env.step(action)
    print(reward)
