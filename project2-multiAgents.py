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
# W3Schools and Stackoverflow for syntax help
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #collect the current score
        points = successorGameState.getScore()

        #calculate food distances
        fooddistances = []
        for food in newFood.asList():
            fooddist = manhattanDistance(newPos, food)
            fooddistances.append(fooddist)

        if fooddistances:
            points = points + (10.0 / min(fooddistances))

        #rewarding pacman
        for i, ghost in enumerate(newGhostStates):
            ghostdist = manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[i] > 0:
                points = points + (10.0 / ghostdist)

        #penalize pacman
        for i, ghost in enumerate(newGhostStates):
            ghostdist = manhattanDistance(newPos, ghost.getPosition())
            if ghostdist <= 1:
                points = points - 1000
            else:
                points = points - (2.0 / ghostdist)

        return points

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentindex):
            #status check for depth and game state
            if any([depth == self.depth, state.isWin(), state.isLose()]):
                return self.evaluationFunction(state)
            
            legalactions = state.getLegalActions(agentindex)

            #maximizing agent
            if agentindex == 0:
                bestpoints = float('-inf')
                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    nextagent = 1
                    points = minimax(nextstate, depth, nextagent)
                    bestpoints = max(bestpoints, points)
                return bestpoints
            
            else:
                #minimizing agent
                nextagent = (agentindex + 1) % state.getNumAgents()
                if nextagent == 0:
                    newdepth = depth + 1
                else:
                    newdepth = depth
                
                bestpoints = float('inf')
                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    points = minimax(nextstate, newdepth, nextagent)
                    bestpoints = min(bestpoints, points)
                return bestpoints

        legalactions = gameState.getLegalActions(0)
        bestpoints = float('-inf')
        
        #evaluation of actions that could be taken
        for action in legalactions:
            nextState = gameState.generateSuccessor(0, action)
            points = minimax(nextState, 0, 1)
            
            values = [bestpoints, points]
            bestpoints = max(values)

            if bestpoints == points:
                bestaction = action
            else:
                bestaction = bestaction

        return bestaction
        util.raiseNotDefined() 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, depth, agentindex, alpha, beta):
            #status check for depth and game state
            if any([depth == self.depth, state.isWin(), state.isLose()]):
                return self.evaluationFunction(state)
            
            legalactions = state.getLegalActions(agentindex)
            
            #maximizing agent
            if agentindex == 0:  
                maxpoints = float('-inf')
                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    nextagent = 1
                    points = alphabeta(nextstate, depth, nextagent, alpha, beta)
                    maxpoints = max(maxpoints, points)
                    alpha = max(alpha, maxpoints)
                    if beta < alpha:  
                        break
                return maxpoints
            
            else:  
                #minimizing agent
                minpoints = float('inf')
                nextagent = (agentindex + 1) % state.getNumAgents()
                if nextagent == 0:
                   nextdepth = depth + 1  
                else:
                   nextdepth = depth

                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    points = alphabeta(nextstate, nextdepth, nextagent, alpha, beta)
                    minpoints = min(minpoints, points)
                    beta = min(beta, minpoints)
                    if beta < alpha:  
                        break
                return minpoints
        
        alpha = float('-inf')
        beta = float('inf')
        bestpoints = float('-inf')
        legalactions = gameState.getLegalActions(0)

        #evaluation of actions that could be taken
        for action in legalactions:
            nextstate = gameState.generateSuccessor(0, action)
            points = alphabeta(nextstate, 0, 1, alpha, beta)
            values = [bestpoints, points]
            bestpoints = max(values)
            if bestpoints == points:
                bestaction = action
            alpha = max(alpha, bestpoints)

        return bestaction
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
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentindex):
            #status check for depth and game state
            if any([depth == self.depth, state.isWin(), state.isLose()]):
                return self.evaluationFunction(state)

            legalactions = state.getLegalActions(agentindex)

            #maximizing agent
            if agentindex == 0:
                bestpoints = float('-inf')
                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    nextagent = 1
                    points = expectimax(nextstate, depth, nextagent)
                    bestpoints = max(bestpoints, points)
                return bestpoints

            else:
                #minimizing agent
                nextagent = (agentindex + 1) % state.getNumAgents()
                if nextagent == 0:
                    nextdepth = depth + 1
                else:
                    nextdepth = depth

                totalpoints = 0
                for action in legalactions:
                    nextstate = state.generateSuccessor(agentindex, action)
                    points = expectimax(nextstate, nextdepth, nextagent)
                    totalpoints = totalpoints + points

                averagepoints = 0
                if legalactions:
                    averagepoints = totalpoints / len(legalactions)
                return averagepoints

        legalactions = gameState.getLegalActions(0)
        bestpoints = float('-inf')

        #decision for best action
        for action in legalactions:
            nextstate = gameState.generateSuccessor(0, action)
            points = expectimax(nextstate, 0, 1)
            values = [bestpoints, points]
            bestpoints = max(values)
            if bestpoints == points:
                bestaction = action

        return bestaction
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <By calculating scores based on distances to food and ghosts, the code assesses 
    Pac-Man's current game state. It rewards approaching the closest food, considers the farthest 
    food distance, and adjusts based on ghost proximity and scared states. A penalty for remaining
    food items is also implemented encouraging pacman to clear the board>
    """
    "*** YOUR CODE HERE ***"

    pacmanpos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghoststates = currentGameState.getGhostStates()
    foodlist = food.asList()
    
    #calculate the distance to food
    fooddistances = []
    for foodpos in foodlist:
        dist = manhattanDistance(pacmanpos, foodpos)
        fooddistances.append(dist)

    #closest food
    if foodlist:
        closestfooddist = min(fooddistances)
    else:
        closestfooddist = 0

    #farthest food
    if foodlist:
        farthestfooddist = max(fooddistances)
    else:
        farthestfooddist = 0
    
    #calculate ghost distance and scared time
    ghostdistances = [manhattanDistance(pacmanpos, ghost.getPosition()) for ghost in ghoststates]
    scaredtimes = [ghostState.scaredTimer for ghostState in ghoststates]
    
    #for nearest ghost distance
    nearestghostdist = min(ghostdistances) if ghoststates else 0
    scaredghosts = [ghostDist for i, ghostDist in enumerate(ghostdistances) if scaredtimes[i] > 0]
    
    remainingfoodcount = len(foodlist)    
    points = currentGameState.getScore()    
    foodpoints = -closestfooddist  
    foodpenalty = farthestfooddist  
    ghostpenalty = 0

    if nearestghostdist < 2:
        ghostpenalty = 500  

    ghostpoints = 0

    for i, ghostdist in enumerate(ghostdistances):
        if scaredtimes[i] > 0 and ghostdist < 3:  
            ghostpoints = ghostpoints + (100 / (ghostdist + 1))

    foodremainingpenalty = remainingfoodcount * 10  
    totalpoints = points + foodpoints - foodpenalty - ghostpenalty + ghostpoints - foodremainingpenalty

    return totalpoints
# Abbreviation
better = betterEvaluationFunction