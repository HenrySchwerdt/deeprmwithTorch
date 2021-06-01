import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import enivronment
import parameters
import pg_network
import numpy as np
import torch as T


pa = parameters.Parameters()
env = enivronment.Env(pa)


agent = pg_network.Agent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=[pa.network_input_height,pa.network_input_width],
                     n_actions=pa.network_output_dim, mem_size=50000, eps_min=0.1,
                     batch_size=100, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='deep-rm')
scores, eps_history = [], [] 

n_games = 500
# TRAINING
for i in range(n_games):
    score = 0
    done = False
    env.reset()
    ob = env.observe()
    while not done:
        action = agent.choose_action(ob)
        ob_, reward, done, info = env.step(action)
        score+= reward
        agent.store_transition(ob,action,reward,ob_,done)
        agent.learn()
        ob = ob_
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores)

    print('episode', i, 'score %.2f' % score,'mean score %.2f' % avg_score, 'epsilon %.2f'  % agent.epsilon)
# EXAMPLE
env2 = enivronment.Env(pa,render=True)
done = False
ob = env.observe()
while not done:
     action = agent.choose_action(ob)
     print('action: ',action)
     ob_, reward, done, info = env2.step(action)

