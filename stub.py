# Imports.
import numpy as np
import numpy.random as npr
import math

from run import ReinforcementLearning, ReinforcementLearningParam

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.learningrateparam = 0.001
        self.discountrateparam = 1.0
        self.alpha = 50
        #self.rl = ReinforcementLearning(2, 3, self.learningrateparam, self.discountrateparam)
        self.rl = ReinforcementLearningParam(2, 3, 6, self.learningrateparam, self.discountrateparam)
        self.epsilon = 0.01
        self.nruns = 150

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        state_cur = self.__basis(state)

        new_action = self.rl.best_move(state_cur)

        if npr.rand() < (self.epsilon*math.exp(-0.01*np.sum(hist))):
             new_action = 0 if (npr.rand() < 0.5) else 1
             #print "====random action taken: " + str(new_action)

        if self.last_reward:
            self.rl.update(self.last_state, self.last_action, self.last_reward, state_cur)

        self.last_action = new_action
        self.last_state  = state_cur

        return self.last_action


    def __basis(self, state):
        phi =  [1] #[1, state['score']] + state['tree'].values() + state['monkey'].values()
        dist_top = min(state['monkey']['top'],400)
        dist_bottom = max(0,state['monkey']['bot'])
        dist_center = dist_bottom - 200
        above_below_switch = 1 if dist_center > 0 else -1
        #phi.append(5.0/(400.1-dist_top))
        #phi.append(5.0/(0.1+dist_bottom))
        phi.append(abs(state['tree']['bot'] - state['monkey']['bot']))
        phi.append(abs(state['tree']['top'] - state['monkey']['top']))
        # phi.append(dist_top)
        # phi.append(dist_bottom)
        # phi.append(1.02 ** abs(dist_center) * above_below_switch)
        #phi.append(state['monkey']['vel'])
        #print(phi)



        phi = []
        phi.append((state['monkey']['bot'] - state['tree']['bot']) < self.alpha)
        phi.append((state['tree']['top'] - state['monkey']['top']) < self.alpha)
        phi.append((state['tree']['dist']) < self.alpha)


        
        return np.array(phi)

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        #print "state: " + str(self.last_state) + " action: " + str(self.last_action)

        if reward < 0:
                #print "====reward: " + str(reward)
                return

        if reward < 0:
                reward = -1
        elif reward > 0:
                reward = 1
        else:
                reward = 0

        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        #print ii
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        if((ii+1)%10==0):
            print "mean after set number %d of 10 is %f!" %((ii+1)/10,np.mean(np.array(hist)))

        # Reset the state of the learner.
        learner.reset()

    print "----------------------------------------"

    # print hist
    # print np.mean(np.array(hist))
    for i in range(0,iters,10):
        print "mean of set number %d of 10 is %f!" %((i+10)/10,np.mean(np.array(hist[i:i+10])))
    
    print "----------------------------------------"

    print "The best iteration was iteration number %d, which got a score of %d!" %(np.argmax(hist), np.max(hist))

    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
        nruns = agent.nruns
	run_games(agent, hist, nruns, 1)

	# Save history. 
	np.save('hist',np.array(hist))


