import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import enivronment
import parameters
import pg_network
import numpy as np
import torch as T
import util


pa = parameters.Parameters()
env = enivronment.Env(pa)

agent = pg_network.Agent(gamma=1, epsilon=1.0, lr=0.00001,
                     input_dims=[pa.network_compact_dim],
                     n_actions=pa.network_output_dim, mem_size=100000, eps_min=0.1,
                     batch_size=64, replace=1000, eps_dec=1e-4,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='deep-rm')

best_score = -np.inf
load_checkpoint = False
n_games = 1000

if load_checkpoint:
        agent.load_models()


fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
figure_file = 'plots/' + fname + '.png'

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    done = False
    env.reset()
    observation = env.observe()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward

        if not load_checkpoint:
            agent.store_transition(observation, action,
                                     reward, observation_, done)
            agent.learn()
        observation = observation_
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    #print('episode: ', i,'score: ', score,
    #         ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
    #        'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    util.plot_learning_curve(steps_array, scores, eps_history, figure_file)
