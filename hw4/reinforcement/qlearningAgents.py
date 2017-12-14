# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    """
    Initialize self.qValues to an empty dict
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = {} # Initialize self.qValues 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        """
        Initialize qValue to 0.0, if the (state, action) pair has been seen before
        then assign its q-value to qValue from self.qValues
        """
        qValue = 0.0  # default value
        if (state, action) in self.qValues: # if the (state, action) pair has been seen before
          qValue = self.qValues[(state, action)]  # assign its q-value to qValue
        return qValue
        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        """
        Return a q-value of 0.0 for a terminal state
                          or
        if the list of actions returned for that state is not empty then, 
        calculate q-value for every such action and return the max of all
        such q-values.
        """
        actions = self.getLegalActions(state) # get a list of all possible actions for the given state
        if len(actions) == 0: # check if its empty (or that the state is a terminal state)
          return 0.0
        else: 
          qValues = set()
          for action in actions: # for every action in list of actions
            qValues.add(self.getQValue(state, action))  # calculate its q-value
          return max(qValues) # return max of all such q-values
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        """
        Return none for a terminal state
                          or
        if the list of actions returned for that state is not empty then, 
        calculate q-value for every such action and return the action with max q-value.
        """
        actions = self.getLegalActions(state) # get a list of all possible actions for the given state
        if len(actions) == 0: # check if its empty (or that the state is a terminal state)
          return None
        else: 
          actionsDict = {}  # create a dict for all possible actions, where an action is the key and its value is the q-value of that action
          for action in actions: # for every such action
            actionsDict[action] = self.getQValue(state, action) # calculate its q-value
          return max(actionsDict, key = actionsDict.get)  # return action with max q-value
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        """
        Use prob self.epsilon to pick a random action if true otherwise pick the action
        with max q-value calculated using computeActionFromQValues(state) function.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)  # get list of all possible actions for given state
        action = None # default value
        prob = self.epsilon
        if util.flipCoin(prob): # (using hint to) pick a random action using probability prob
          action = random.choice(legalActions)  # (using hint to) pick a random action
        else: 
          action = self.computeActionFromQValues(state) # otherwise choose the action with max q-value calculated using computeActionFromQValues(state) function
        return action
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        """
        update q-values using the foll. formulas: 
          1. sample <- reward(action, state, nextState) + gamma * v(nextState)
          2. v(state) <- v(state) + alpha (sample - v(state))
        
        v(nextState) is calculated using function computeValueFromQValues(state) 
        v(state) is calculated using function getQValue(state, action)
        """
        gamma = self.discount
        alpha = self.alpha

        qValue = self.getQValue(state, action)  # get q-value of current state using getQValue(state, action) function
        nextQValue = self.computeValueFromQValues(nextState)  # get q-value of next state using computeActionFromQValues(state) function
        sample = reward + gamma * nextQValue  # calculate sample using formula: reward(action, state, nextState) + gamma * v(nextState)

        self.qValues[(state, action)] = qValue + alpha * (sample - qValue)  # update q-values using formula: v(state) <- v(state) + alpha (sample - v(state))
        # the formulas are given in the reinforcement learning slides under temporal diff learning

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        """
        calculate q-values for feature i using formula: Q(s,a) = sigma fi(s,a) * wi
        """
        qValue = 0.0  # initial value
        featureVector = self.featExtractor.getFeatures(state, action) # get list of features
        for feature in featureVector: # for each feature calculate weight(feature) * value(feature)
          qValue += self.weights[feature] * featureVector[feature]  # assign q-value as the sum of (weight(feature) * value(feature)) for all features
        return qValue
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        """
        update weight of feature i using formula: 
          wi = wi + alpha * diff * fi(s,a)
          diff = (r + gamma * max over a' (Q(s', a')) - Q(s, a))
        """
        gamma = self.discount
        qValue = self.getQValue(state, action)  # get Q(s, a)
        nextQValue = self.computeValueFromQValues(nextState)  # get max over a' (Q(s', a')) i.e. calculate q-value for every action amongst all possible actions in the next state and return the max q-value
        diff = (reward + gamma * nextQValue) - qValue   # calculate diff using formula: diff = (r + gamma * max over a' (Q(s', a')) - Q(s, a))

        featureVector = self.featExtractor.getFeatures(state, action) # get list of all features
        for feature in featureVector: # for every feature 
          self.weights[feature] = self.weights[feature] + (self.alpha * diff * featureVector[feature])  # update its weight using formula: wi = wi + alpha * diff * fi(s,a)
        # the formulas are provided as a part of the assignment in q8

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
