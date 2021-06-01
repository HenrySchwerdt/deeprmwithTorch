import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import enivronment
import parameters
import pg_network
import numpy as np
import torch as T


pa = parameters.Parameters()
env = enivronment.Env(pa)

agent = pg_network.Agent(gamma=0.99, epsilon=1.0,batch_size=10,n_actions=pa.network_output_dim,
                input_dims=[pa.network_compact_dim],eps_end=0.01, lr=0.003)
scores, eps_history = [], [] 

n_games = 500

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
    scores.append(scores)
    eps_history.append(agent.epsilon)
    #avg_score = np.mean(*scores)

    print('episode', i, 'score %.2f' % score,'mean score %.2f' % 1, 'epsilon %.2f'  % agent.epsilon)

env2 = enivronment.Env(pa,render=True)
done = False
ob = env.observe()
while not done:
     action = agent.choose_action(ob)
     ob_, reward, done, info = env2.step(action)

