# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        scores = set()
        food = currentGameState.getFood()
        foodList = food.asList()

        for foodPos in foodList: 
          score = -10
          foodDis = manhattanDistance(foodPos, newPos)
          score *= foodDis
          score += ghostsEvaluationFunction(newPos, newGhostStates)
          scores.add(score)

        return max(scores)

def ghostsEvaluationFunction(pos, ghostStates): 
  score = 0
  for ghost in ghostStates: 
    ghostPos = ghost.getPosition()
    ghostDis = manhattanDistance(pos, ghostPos)
    if ghostDis > 1: 
      score += ghostDis
    else: 
      score -= 1000        

    if ghost.scaredTimer != 0: 
      score += 100
  return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        # a ply consists of one pacman move (max) and all ghosts' responses (mins) 
        # agents - 1 = # of times min is called (i.e. #ghosts) after max (i.e. pacman move)

        agents = gameState.getNumAgents()
        
        def minimax(gameState, agentIndex, depth): 
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (None, self.evaluationFunction(gameState))

          if agentIndex != agents: 
            return minimum(gameState, agentIndex, depth)
          else: 
            return maximum(gameState, self.initialAgentIndex, depth)   # reset agentIndex in subsequent max calls

        def minimum(gameState, agentIndex, depth): 
          dic = {}
          v = float("inf")
          actions = gameState.getLegalActions(agentIndex)
          for action in actions: 
            successor = gameState.generateSuccessor(agentIndex, action)
            # increment depth BEFORE call to minimax if...
            if agentIndex + 1 == agents:  # next call in minimax is max
              newV = minimax(successor, agentIndex + 1, depth + 1)[1]  
            else: 
              newV = minimax(successor, agentIndex + 1, depth)[1]
            if newV < v: 
              v = newV
              dic[v] = action
          return (dic[v], v)

        def maximum(gameState, agentIndex, depth): 
          dic = {}
          v = float("-inf")
          actions = gameState.getLegalActions(agentIndex)
          for action in actions: 
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = minimax(successor, agentIndex + 1, depth)[1]   # 1 references value in tuple (action, value[action])
            if newV > v: 
              v = newV
              dic[v] = action
          return (dic[v], v)

        self.initialAgentIndex, self.initialDepth = 0, 0
        return maximum(gameState, self.initialAgentIndex, self.initialDepth)[0]   # initial call to max (pacman is always agent 0)
        # 0 references action where
        # minimax, max and min return a tuple (action, value[action])
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agents = gameState.getNumAgents()

        def minimax(gameState, agentIndex, depth, alpha, beta): 
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (None, self.evaluationFunction(gameState))

          if agentIndex != agents: 
            return minimum(gameState, agentIndex, depth, alpha, beta)
          else: 
            return maximum(gameState, self.initialAgentIndex, depth, alpha, beta)  

        def minimum(gameState, agentIndex, depth, alpha, beta): 
          dic = {}
          v = float("inf")
          actions = gameState.getLegalActions(agentIndex)
          for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == agents:
              newV = minimax(successor, agentIndex + 1, depth + 1, alpha, beta)[1]  
            else: 
              newV = minimax(successor, agentIndex + 1, depth, alpha, beta)[1]
            if newV < v: 
              v = newV
              dic[v] = action
            if v < alpha: 
              return (dic[v], v)
            beta = min(beta, v)
          return (dic[v], v)

        def maximum(gameState, agentIndex, depth, alpha, beta): 
          dic = {}
          v = float("-inf")
          actions = gameState.getLegalActions(agentIndex)
          for action in actions: 
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = minimax(successor, agentIndex + 1, depth, alpha, beta)[1]  
            if newV > v: 
              v = newV
              dic[v] = action
            if v > beta: 
              return (dic[v], v)
            alpha = max(alpha, v)
          return (dic[v], v)

        self.initialAgentIndex, self.initialDepth = 0, 0
        alpha, beta = float("-inf"), float("inf")
        return maximum(gameState, self.initialAgentIndex, self.initialDepth, alpha, beta)[0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        agents = gameState.getNumAgents()
        
        def minimax(gameState, agentIndex, depth): 
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (None, self.evaluationFunction(gameState))

          if agentIndex != agents: 
            return expected(gameState, agentIndex, depth)
          else: 
            return maximum(gameState, self.initialAgentIndex, depth) 

        def expected(gameState, agentIndex, depth): 
          dic = {}
          v = 0
          actions = gameState.getLegalActions(agentIndex)
          p = 1.0/len(actions)
          for action in actions: 
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == agents:  
              newV = minimax(successor, agentIndex + 1, depth + 1)[1]  
            else: 
              newV = minimax(successor, agentIndex + 1, depth)[1]
            v += (p * newV)
            dic[v] = action
          return (dic[v], v)

        def maximum(gameState, agentIndex, depth): 
          dic = {}
          v = float("-inf")
          actions = gameState.getLegalActions(agentIndex)
          for action in actions: 
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = minimax(successor, agentIndex + 1, depth)[1]   
            if newV > v: 
              v = newV
              dic[v] = action
          return (dic[v], v)

        self.initialAgentIndex, self.initialDepth = 0, 0
        return maximum(gameState, self.initialAgentIndex, self.initialDepth)[0] 
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The betterEvaluationFunction calculates the distance 
      between pacman position and the closest food. It subtracts this distance 
      from the score. It also calculates the scared timer for the ghost. If the ghost
      is scared, it adds 1000 to score. Finally, when the len(foodList) is 0 
      (i.e. terminal state), it returns infinity.
    """

    pacmanPos = currentGameState.getPacmanPosition()

    foodList = currentGameState.getFood().asList()
    if len(foodList) == 0: 
        return float("inf")

    closestFoodDis = float("inf")
    for foodPos in foodList: 
      foodDis = manhattanDistance(foodPos, pacmanPos)
      closestFoodDis = min(closestFoodDis, foodDis)

    ghostState = currentGameState.getGhostStates()[0]   # only 1 ghost
    scared = ghostState.scaredTimer

    score = currentGameState.getScore()

    if scared != 0: 
      return score + 1000

    score -= closestFoodDis
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

