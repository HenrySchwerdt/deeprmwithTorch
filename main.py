import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import enivronment
import parameters
import pg_network
import torch


pa = parameters.Parameters()
print(pa)
env = enivronment.Env(pa)
indim = env.observe().size

dqn = pg_network.DQN(indim,pa.network_output_dim)
print(dqn.forward(torch.tensor(env.observe())))