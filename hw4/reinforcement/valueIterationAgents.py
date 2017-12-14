# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        """
        Run value iteration self.iteration times for every mdp state
        to calculate its q-value. The q-value of a non-terminal state is calculated 
        by getting a list of all possible actions for that state and then taking the max
        of q-values calculated (using computeQValueFromValues(state, action)) for every such 
        action.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        self.previousValues = util.Counter()  # store previously calculated q-values
        for i in range(self.iterations):  # run value iteration, self.iterations times
          for state in self.mdp.getStates():  # for every mdp state
            if not self.mdp.isTerminal(state):  # if that state is not a terminal state (q-value of a terminal state will be 0 by default in self.values)
              actions = set() # get its list of all possible actions 
              for action in self.mdp.getPossibleActions(state): # for every such action 
                actions.add(self.computeQValueFromValues(state, action))  # calculate its q-value
              self.values[state] = max(actions) # and assign the max of all such q-values as q-value of current state
          self.previousValues = self.values.copy()  # update previousValues since it's used for computing q-values in computeQValueFromValues
        self.previousValues = self.values.copy()  # update previousValues after last iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        """
        Calculate q-value for a (state, action) pair by 
          1. getting its list of (nextState, prob) pairs
          2. getting reward for every (newState, prob) pair in that list
          3. and using the foll. formula: 
              sigma (prob(state, action, newState) * [reward(state, action, newState) + gamma * value(newState)])
        
        """
        qValue = 0  # initial q-value
        for nextStateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):  # for every (nextState, prob) pair in list of (nextState, prob) pairs
          nextState, prob = nextStateAndProb[0], nextStateAndProb[1]  
          reward = self.mdp.getReward(state, action, nextState) # get its reward
          qValue += prob * (reward + self.discount * self.previousValues[nextState])  # calculate q-value using formula: sigma (prob(state, action, newState) * [reward(state, action, newState) + gamma * value(newState)])
        return qValue     
        # the formulas are given in the reinforcement learning slides under value iteration
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        """
        Get a list of all possible actions for a non-terminal state and return the action
        with max q-value calculated by calling computeQValueFromValues(state, action) function
                    or
        return none for terminal state.
        """
        if not self.mdp.isTerminal(state): # if state is not a terminal state
          actions = {}  # create a dict for all possible actions in that state, where an action is the key and its value is the q-value of that action
          for action in self.mdp.getPossibleActions(state): # for every such action
            actions[action] = self.computeQValueFromValues(state, action) # calculate its q-value
          return max(actions, key = actions.get)  # return the action with max q-value
        else: 
          return None   # return none for a terminal state
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
