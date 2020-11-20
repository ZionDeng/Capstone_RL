# #  super simple example
# initialize Q(s,a) arbitrarily
# repeat(for each):
#     initialize s
#     repeat (for each step):
#         choose a from s using policy derived from Q
#         Take action a, observe r, s'
#         Q(s,a) = Q(s,a) + alpha[r + lambda * max(Q(s', a') - Q (s,a))]
#         s = s'
#     until s is terminal

import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
LAMBDA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        # random choose when act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES - 2:
            state_next = 'terminal'
            reward = 1
        else:
            state_next = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_next = state
        else:
            state_next = state - 1
    return state_next, reward


def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total steps = %s' % (
            episode + 1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(1)
        print('\r')

    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminate = False
        update_env(state, episode, step_counter)
        while not is_terminate:
            action = choose_action(state, q_table)
            state_next, reward = get_env_feedback(state, action)
            q_predict = q_table.ix[state, action]
            if state_next is not 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[state_next, :].max()
            else:
                q_target = reward
                is_terminate = True

            q_table.ix[state, action] += ALPHA * (q_target - q_predict)
            state = state_next

            update_env(state, episode, step_counter + 1)
            step_counter += 1
        print(q_table)
        
    return q_table


if __name__ == "__main__":
    Q_table = rl()
    print('\r\n Q-table: \n')
    print(Q_table)
