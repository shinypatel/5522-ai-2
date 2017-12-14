# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
    
s = Directions.SOUTH
w = Directions.WEST
n = Directions.NORTH
e = Directions.EAST

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    return  [s, s, w, s, w, w, s, w]

def initialize(problem): 
    return problem.getStartState(), None, {}, set(), set()

def directions(action): 
    if action == 'East': action = e
    elif action == 'West': action = w
    elif action == 'North': action = n
    elif action == 'South': action = s  
    return action  

def actions(startState, state, parentDic): 
    actions = []
    while state != startState:  #traverse the path from goalState to startState
        action = parentDic[state][1]
        actions.insert(0, action)
        state = parentDic[state][0]
    return actions

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    startState, state, parentDic, fringe, visited = initialize(problem)

    stack = util.Stack()

    stack.push(startState)    
    while not stack.isEmpty(): 
        state = stack.pop()
        if state not in visited: 
            visited.add(state)
        else: 
            continue
        if problem.isGoalState(state): 
            break
        successors = problem.getSuccessors(state)
        for successor in successors: 
            successorState = successor[0]
            action = successor[1]
            action = directions(action)
            if not (successorState, state) in fringe: #check for bi-directional nodes in graph i.e A->B and B->A
                fringe.add((state, successorState))
                stack.push(successorState)
                parentDic[successorState] = (state, action) #replace value of existing keys if any
                #dfs path uses the last successorState -> (state, action) key-value pair               
    return actions(startState, state, parentDic)
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    startState, state, parentDic, fringe, visited = initialize(problem)

    queue = util.Queue()
    
    queue.push(startState)    
    while not queue.isEmpty(): 
        state = queue.pop()
        if state not in visited: 
            visited.add(state)
        else: 
            continue
        if problem.isGoalState(state): 
            break
        successors = problem.getSuccessors(state)
        for successor in successors:  
            successorState, action = successor[0], successor[1]
            action=directions(action)
            if not (successorState, state) in fringe: 
                fringe.add((state, successorState))
                queue.push(successorState)
                if successorState not in parentDic: 
                    parentDic[successorState] = (state, action)
                    #bfs path uses the first successorState -> (state, action) key-value pair               
    return actions(startState, state, parentDic)
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    startState, state, parentDic, fringe, visited = initialize(problem)

    queue = util.PriorityQueue()
    
    queue.push((startState, 0), 0)    
    while not queue.isEmpty(): 
        el = queue.pop()
        state, pathCost = el[0], el[1]
        if state not in visited: 
            visited.add(state)
        else: 
            continue
        if problem.isGoalState(state): 
            break
        successors = problem.getSuccessors(state)
        for successor in successors: 
            successorState, action = successor[0], successor[1]
            successorPathCost = successor[2] + pathCost
            action = directions(action)
            if not (successorState, state) in fringe: 
                fringe.add((state, successorState))
                queue.push((successorState, successorPathCost), successorPathCost)
                if successorState not in parentDic: 
                    parentDic[successorState] = (state, action, successorPathCost)
                else: 
                    oldPathCost = parentDic[successorState][2]
                    if successorPathCost < oldPathCost: 
                        parentDic[successorState] = (state, action, successorPathCost)
    return actions(startState, state, parentDic)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startState, state, parentDic, fringe, visited = initialize(problem)

    queue = util.PriorityQueue()
    
    queue.push((startState, 0, 0), 0)    
    while not queue.isEmpty(): 
        el = queue.pop()
        state, gOfN, fOfN = el[0], el[1], el[2]
        if state not in visited: 
            visited.add(state)
        else: 
            continue
        if problem.isGoalState(state): 
            break
        successors = problem.getSuccessors(state)
        for successor in successors: 
            successorState, action = successor[0], successor[1]
            successorGOfN = successor[2] + gOfN
            successorFOfN = successorGOfN + heuristic(successorState, problem)
            action = directions(action)
            if not (successorState, state) in fringe: 
                fringe.add((state, successorState))
                queue.push((successorState, successorGOfN, successorFOfN), successorFOfN)
                if successorState not in parentDic: 
                    parentDic[successorState] = (state, action, successorFOfN)
                else: 
                    oldFOfN = parentDic[successorState][2]
                    if successorFOfN < oldFOfN: 
                        parentDic[successorState] = (state, action, successorFOfN)
    return actions(startState, state, parentDic)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
