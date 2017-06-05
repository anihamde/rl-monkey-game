import numpy as np


class ReinforcementLearning(object):

	def __init__(self, num_actions, state_dimensions, learning_rate, discount_factor):
		self.num_actions = num_actions
		self.state_dimensions = state_dimensions
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.W = np.zeros((self.num_actions, self.state_dimensions))



	def best_move(self, s):
                #print self.W
		return np.argmax([self.__eval_Q(s, a, self.W[a]) for a in range(0, self.num_actions)])


	def update(self, s, a, r, s_next):
		print(a)
		self.W[a]
		print(self.W[1])
		print(self.W[0])
		gradient_a = self.__eval_Q_loss_gradient(s, a, r, s_next, self.W[a])		
		step = gradient_a * self.learning_rate
		self.W[a] = self.W[a] - step


	def __eval_Q_loss_gradient(self, s, a, r, s_next, w_a):
		w_old = w_a # TODO: Really confused what w_old is supposed to be so just assuming it's the same as w for now
		q = self.__eval_Q(s, a, w_a)
		q_next = max([self.__eval_Q(s_next, a_next, self.W[a_next]) for a_next in range(0, self.num_actions)])
		q_eqiv = r + self.discount_factor*q_next
		q_gradient = s
		print(q_gradient)
		return (q_next - q_eqiv)*q_gradient


	def __eval_Q(self, s, a, w_a):
		return np.dot(w_a, s)

class ReinforcementLearningParam(object):

	def __init__(self, num_actions, state_dimensions, num_states, learning_rate, discount_factor):
		self.num_actions = num_actions
		self.state_dimensions = state_dimensions
		self.num_states = num_states
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.W = np.zeros((self.num_states, self.num_actions))
		self.states = np.zeros((self.num_states, self.state_dimensions))
		self.last_state = -1

	def best_move(self, s_raw):
                s = self.__get_state(s_raw)
		return np.argmax([self.__eval_Q(s, a, self.W) for a in range(0, self.num_actions)])

	def update(self, s_raw, a, r, s_next_raw):
                s = self.__get_state(s_raw)
                s_next = self.__get_state(s_next_raw)
		gradient_a = self.__eval_Q_loss_gradient(s, a, r, s_next, self.W)		
		step = gradient_a * self.learning_rate
		self.W[s, a] = self.W[s, a] - step

	def __eval_Q_loss_gradient(self, s, a, r, s_next, w_a):
		q = self.__eval_Q(s, a, w_a)
		q_next = max([self.__eval_Q(s_next, a_next, self.W) for a_next in range(0, self.num_actions)])
		q_eqiv = r + self.discount_factor * q_next
		return q - q_eqiv

	def __eval_Q(self, s, a, w_a):
		return w_a[s, a]

	def __get_state(self, s):
                s_idx = np.where(np.all(self.states == s,axis=1))[0]
                if len(s_idx) == 0:
                        self.last_state += 1
                        s_idx = self.last_state
                        self.states[s_idx,:] = s
                        return s_idx
                else:
                        return s_idx[0]
