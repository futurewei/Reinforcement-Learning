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

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for itera in range(self.iterations):
           tempVal=self.values.copy()
           for state in self.mdp.getStates():
              if self.mdp.isTerminal(state):
                   continue
              bestAction=self.computeActionFromValues(state)
              val=self.computeQValueFromValues(state, bestAction)
              tempVal[state]=val
           self.values=tempVal


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
        Qvalue=0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState=transition[0]
            prob=transition[1]
            Qvalue+=prob*(self.mdp.getReward(state, action, nextState)+self.discount*self.getValue(nextState))
        return Qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction=None
        best=-float('inf')
        for action in self.mdp.getPossibleActions(state):
            Qvalue=self.computeQValueFromValues(state, action)
            if Qvalue>best:
                best=Qvalue  
                bestAction=action
        return bestAction

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
        for i in range(self.iterations):
            state=self.mdp.getStates()[i % len(self.mdp.getStates())]
            if self.mdp.isTerminal(state):
                continue
            bestAction=self.computeActionFromValues(state)
            val=self.computeQValueFromValues(state, bestAction)
            self.values[state]=val

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
        queue=util.PriorityQueue()
        predecessors={}
        for s in self.mdp.getStates():
            predecessors[s] = set()

        for pred in self.mdp.getStates():
          actions=self.mdp.getPossibleActions(pred)
          for action in actions:
              transitionList=self.mdp.getTransitionStatesAndProbs(pred, action)
              for succ in self.mdp.getStates():
                  for state in transitionList:
                    if state[0]==succ and state[1]>0:
                      predecessors[succ].add(pred)
                      break

        for non_terminal in self.mdp.getStates():
           if self.mdp.isTerminal(non_terminal):
              continue
           best_action=self.computeActionFromValues(non_terminal)
           highQ=self.computeQValueFromValues(non_terminal, best_action)
           diff=abs(self.values[non_terminal]-highQ)
           queue.push(non_terminal,-diff)

        for i in range(self.iterations):
          if queue.isEmpty():
             break
          else:
             s=queue.pop()
             best_action=self.computeActionFromValues(s)
             best_Q=self.computeQValueFromValues(s, best_action)
             self.values[s]=best_Q
             for p in predecessors[s]:
                 bestAct=self.computeActionFromValues(p)
                 higQ=self.computeQValueFromValues(p, bestAct)
                 diff=abs(self.values[p]-higQ)
                 if diff>self.theta:
                    queue.update(p, -diff)