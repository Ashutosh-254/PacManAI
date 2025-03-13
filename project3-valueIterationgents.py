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
import collections


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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    #performing value iteration for fixed iterations
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
            updatedvalues = util.Counter()  
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    updatedvalues[state] = 0
                    continue

                maxvalue = float('-inf')
                possibleactions = self.mdp.getPossibleActions(state)
                for action in possibleactions:
                    qvalue = self.computeQValueFromValues(state, action)
                    if qvalue > maxvalue:
                        maxvalue = qvalue  

                if maxvalue == float('-inf'):
                    updatedvalues[state] = 0  
                else:
                    updatedvalues[state] = maxvalue  

            self.values = updatedvalues 

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
        "*** YOUR CODE HERE ***"
        totalvalue = 0
        #calculation of qvalue
        for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextstate)
            totalvalue = totalvalue + prob * (reward + self.discount * self.values[nextstate])
        return totalvalue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        bestaction = None
        maxactionvalue = float('-inf')

        #finding action with highest qvalue

        for action in self.mdp.getPossibleActions(state):
            qvalue = self.computeQValueFromValues(state, action)
            if qvalue > maxactionvalue:
                maxactionvalue = qvalue
                bestaction = action

        return bestaction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #determine total number of states
        states = self.mdp.getStates()  
        numstates = len(states)

        for iteration in range(self.iterations):
            state = states[iteration % numstates]  

            if not self.mdp.isTerminal(state):
                bestvalue = float('-inf')
                #finding max value from state
                for action in self.mdp.getPossibleActions(state):
                    qvalue = 0
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextstate, prob in transitions:
                        reward = self.mdp.getReward(state, action, nextstate)
                        qvalue = qvalue + prob * (reward + self.discount * self.values[nextstate])
                    
                    if qvalue > bestvalue:
                        bestvalue = qvalue

                self.values[state] = bestvalue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #state to predecessor connection
        statepredecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        if nextstate not in statepredecessors:
                            statepredecessors[nextstate] = set()
                        statepredecessors[nextstate].add(state)

        #prriority que set up
        statepriorityqueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxqvalue = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    qvalue = self.computeQValueFromValues(state, action)
                    if qvalue > maxqvalue:
                        maxqvalue = qvalue
                statedifference = abs(self.values[state] - maxqvalue)
                statepriorityqueue.update(state, -statedifference)

        #state value update as per priority
        for iteration in range(self.iterations):
            if statepriorityqueue.isEmpty():
                break
            state = statepriorityqueue.pop()

            if not self.mdp.isTerminal(state):
                maxqvalue = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    qvalue = self.computeQValueFromValues(state, action)
                    if qvalue > maxqvalue:
                        maxqvalue = qvalue
                self.values[state] = maxqvalue

            #predecessor update as per priority
            if state in statepredecessors:
                for predecessor in statepredecessors[state]:
                    if not self.mdp.isTerminal(predecessor):
                        maxqvalue = float('-inf')
                        for action in self.mdp.getPossibleActions(predecessor):
                            qvalue = self.computeQValueFromValues(predecessor, action)
                            if qvalue > maxqvalue:
                                maxqvalue = qvalue
                        statedifference = abs(self.values[predecessor] - maxqvalue)
                        if statedifference > self.theta:
                            statepriorityqueue.update(predecessor, -statedifference)