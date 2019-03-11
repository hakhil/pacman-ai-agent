# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from distanceCalculator import Distancer

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    # Offense registration
    self.distancer = Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()
    self.start = gameState.getAgentPosition(self.index)
    self.entryPoints = []
    self.otherPath = False
    self.path = []


    # Defense variables
    self.defense = False
    width = gameState.data.layout.width
    if gameState.isOnRedTeam(self.index):
        width = (width / 2) - 1
    else:
        width = (width / 2)

    # Store the entry points
    for y in range(gameState.getWalls().height):
        if not gameState.hasWall(int(width), y):
            self.entryPoints.append((int(width), y))

    # Inference variables
    self.team = self.getTeam(gameState)
    self.enemy = self.getOpponents(gameState)
    self.possiblePos = []
    for p in gameState.getWalls().asList(False):
        self.possiblePos.append(p)

  # def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    # Update information for the agent
    # Assign some weight to this action and consider it when deciding to
    # select the best action(s) below
    # start = time.time()

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)

    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    return random.choice(bestActions)

  def evaluate(self, gameState, action):
    self.features = self.getFeatures(gameState, action)
    self.weights = self.getWeights(gameState, action)
    return self.features * self.weights

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def updateInferenceByTime(self, enemy, gameState):
      newInference = util.Counter()
      # get old possible pos
      oldPositions = []
      for pos in self.possiblePos:
          if self.inference[enemy][pos] > 0:
              oldPositions.append(pos)

      for oldpos in oldPositions:
          dx = [0, 0, -1, 1, 0]
          dy = [1, -1, 0, 0, 0]

          # find all possible Positions
          possiblePositions = []
          for i in range(len(dx)):
              newPos = (oldpos[0] + dx[i], oldpos[1] + dy[i])
              if newPos in self.possiblePos:
                  possiblePositions.append(newPos)

          possibility = 1.0 / len(possiblePositions)
          for pos in possiblePositions:
              newInference[pos] = possibility * self.inference[enemy][oldpos] + newInference[pos]

      # normalize and update the inference
      newInference.normalize()
      self.inference[enemy] = newInference

  # update the inference by observation
  def updateInferenceByObserve(self, enemy, observation, gameState):
      myPos = gameState.getAgentPosition(self.index)
      noisyDistance = observation[enemy]
      newInference = util.Counter()

      # enemy can be observed by agent
      if gameState.getAgentPosition(enemy) != None:
          for pos in self.possiblePos:
              if pos == gameState.getAgentPosition(enemy):
                  newInference[pos] = 1
              else:
                  newInference[pos] = self.inference[enemy][pos]
          newInference.normalize()
          self.inference[enemy] = newInference
      # unobservable
      else:
          for pos in self.possiblePos:
              if util.manhattanDistance(myPos, pos) <= 5:
                  newInference[pos] = 0
              else:
                  condProb = gameState.getDistanceProb(util.manhattanDistance(myPos, pos), noisyDistance)
                  newInference[pos] = self.inference[enemy][pos] * condProb
          newInference.normalize()
          self.inference[enemy] = newInference

  def positionOnMySide(self, gameState, position):
    if gameState.isOnRedTeam(self.index):
        if position[0] < (gameState.data.layout.width / 2) + 1:
            return True
    else:
        if position[0] < (gameState.data.layout.width / 2):
            return True
    return False

class OffensiveAgent(DummyAgent):

  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)
    self.inference = {}
    for enemy in self.enemy:
        self.inference[enemy] = util.Counter()
        # print self.inference[enemy]
        for pos in self.possiblePos:
            self.inference[enemy][pos] = 0
        self.inference[enemy][gameState.getInitialAgentPosition(enemy)] = 1

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    # Update information for the agent
    # Assign some weight to this action and consider it when deciding to
    # select the best action(s) below
    # start = time.time()
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    myPos = (int(myPos[0]), int(myPos[1]))

    # update the inference table
    # for enemy in self.enemy:
    #     self.updateInferenceByTime(enemy, gameState)
    #     observation = gameState.getAgentDistances()
    #     self.updateInferenceByObserve(enemy, observation, gameState)
    #
    # enemyPos = self.inference[1].argMax()
    # path = bfs(self.index, gameState, self, enemyPos, [])
    # self.debugDraw(path, (1.0, 1.0, 1.0), True)

    opponents = self.getOpponents(gameState)
    dists = gameState.getAgentDistances()

    # Closest ghost
    m = float('inf')
    i = -1
    for opponent in opponents:
        if dists[opponent] < m and gameState.getAgentPosition(opponent) != None:
            m = self.getMazeDistance(myPos, gameState.getAgentPosition(opponent))
            i = opponent

    invDirections = {
        (0, -1): 'South',
        (-1, 0): 'West',
        (1, 0): 'East',
        (0, 1): 'North'
    }

    if myState.isPacman:
        self.otherPath = False
        self.path = []

    if myState.isPacman and m <= 5 and i != -1 and (not gameState.getAgentState(i).isPacman) and gameState.getAgentState(i).scaredTimer < 5:
        opponentPos = gameState.getAgentPosition(i)
        avoid = {opponentPos,
                 (opponentPos[0] - 1, opponentPos[1]),
                 (opponentPos[0] + 1, opponentPos[1])}

        path = bfs(self.index, gameState, self, None, avoid)
        # self.debugDraw(path, (1.0, 1.0, 1.0), True)

        if len(path) > 0:
            return invDirections[(path[0][0] - myPos[0], path[0][1] - myPos[1])]

    elif (not myState.isPacman and i != -1 and len(self.entryPoints) > 0) or self.otherPath:
        path = self.path
        # self.debugDraw(path, (1.0, 1.0, 1.0), True)
        if len(path) == 0 and not self.otherPath:
            end = random.choice(self.entryPoints)
            # if gameState.isOnRedTeam(self.index):
            #     end = (end[0] + 1, end[1])
            # else:
            #     end = (end[0] - 1, end[1])

            opponentPos = gameState.getAgentPosition(i)
            avoid = {opponentPos,
                     (opponentPos[0] - 1, opponentPos[1]),
                     (opponentPos[0] + 1, opponentPos[1])}

            if gameState.isOnRedTeam(self.index):
                for pos in self.entryPoints:
                    avoid.add((pos[0] + 1, pos[1]))
            else:
                for pos in self.entryPoints:
                    avoid.add((pos[0] - 1, pos[1]))

            path = bfs(self.index, gameState, self, end, avoid)
            # self.debugDraw(path, (1.0, 1.0, 1.0), True)
            self.otherPath = True
            if len(path) > 0:
                pos = path.pop(0)
                pos = (int(pos[0]), int(pos[1]))
                self.path = path
                if (pos[0] - myPos[0], pos[1] - myPos[1]) in invDirections.keys():
                    return invDirections[(pos[0] - myPos[0], pos[1] - myPos[1])]
        if len(path) > 0 and self.otherPath:
            pos = path.pop(0)
            pos = (int(pos[0]), int(pos[1]))
            if (pos[0] - myPos[0], pos[1] - myPos[1]) in invDirections.keys():
                return invDirections[(pos[0] - myPos[0], pos[1] - myPos[1])]

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)

    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    action = random.choice(bestActions)
    if len(bestActions) == 1 and action == 'Stop':
        return random.choice(actions)

    return random.choice(bestActions)

  def getFeatures(self, gameState, action):

    features = util.Counter()

    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    currentFood = self.getFood(gameState).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)
    myState = gameState.getAgentState(self.index)

    ghosts = []
    if myState.isPacman:
        opponents = self.getOpponents(gameState)
        agentDistances = gameState.getAgentDistances()
        ghosts = [agentDistances[opponent] for opponent in opponents if
                  (not gameState.getAgentState(opponent).isPacman)]

    # Compute distance to the nearest food

    if len(currentFood) > 2:  # This should always be True, but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in currentFood])
        features['distanceToFood'] = minDistance

        if len(ghosts) > 0:
            features['ghostDist'] = min(ghosts)

    else:
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        features['distHome'] = dist

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'distHome': -10, 'ghostDist': 30}

class DefensiveAgent(DummyAgent):

  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)
    self.inference = {}
    for enemy in self.enemy:
        self.inference[enemy] = util.Counter()
        # print self.inference[enemy]
        for pos in self.possiblePos:
            self.inference[enemy][pos] = 0
        self.inference[enemy][gameState.getInitialAgentPosition(enemy)] = 1

  def chooseAction(self, gameState):

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    myPos = (int(myPos[0]), int(myPos[1]))

    # update the inference table
    for enemy in self.enemy:
        self.updateInferenceByTime(enemy, gameState)
        observation = gameState.getAgentDistances()
        self.updateInferenceByObserve(enemy, observation, gameState)

        # print str(enemy) + ":" + str(enemyPos)

    opponents = self.getOpponents(gameState)
    dists = gameState.getAgentDistances()
    # Closest ghost
    m = float('inf')
    i = -1
    for opponent in opponents:
        if dists[opponent] < m and gameState.getAgentPosition(opponent) != None:
            m = self.getMazeDistance(myPos, gameState.getAgentPosition(opponent))
            i = opponent

    invDirections = {
        (0, -1): 'South',
        (-1, 0): 'West',
        (1, 0): 'East',
        (0, 1): 'North'
    }

    path = self.path
    if i != -1 and gameState.getAgentState(i).isPacman:
        self.otherPath = False

        avoid = set()
        if gameState.isOnRedTeam(self.index):
            for pos in self.entryPoints:
                avoid.add((pos[0] + 1, pos[1]))
        else:
            for pos in self.entryPoints:
                avoid.add((pos[0] - 1, pos[1]))

        path = bfs(self.index, gameState, self, gameState.getAgentPosition(i), avoid)
        if len(path) > 0:
            pos = path[0]
            return invDirections[(pos[0] - myPos[0], pos[1] - myPos[1])]

    elif (not gameState.getAgentState(i).isPacman and len(self.entryPoints) > 0 and not self.otherPath) or (len(self.path) == 0 and self.otherPath):

        m = float('inf')
        end = myPos
        for enemy in self.enemy:
            enemyPos = self.inference[enemy].argMax()
            if self.getMazeDistance(myPos, enemyPos) < m:
                m = self.getMazeDistance(myPos, enemyPos)
                end = enemyPos

        if self.positionOnMySide(gameState, end):
            if gameState.isOnRedTeam(self.index):
                if end[0] > (gameState.data.layout.width / 2) - 1:
                    end = self.entryPoints[0]
                    m = float('inf')
                    for pos in self.entryPoints:
                        if self.getMazeDistance(myPos, pos) < m:
                            m = self.getMazeDistance(myPos, pos)
                            end = pos
                end = (end[0] - 1, end[1])
            else:
                if end[0] < (gameState.data.layout.width / 2):
                    end = self.entryPoints[0]
                    m = float('inf')
                    for pos in self.entryPoints:
                        if self.getMazeDistance(myPos, pos) < m:
                            m = self.getMazeDistance(myPos, pos)
                            end = pos
                end = (end[0] + 1, end[1])

            enemyPos = self.inference[1].argMax()
            path = bfs(self.index, gameState, self, enemyPos, [])
            self.debugDraw(path, (1.0, 1.0, 1.0), True)
        else:
            end = random.choice(self.entryPoints)

            if gameState.isOnRedTeam(self.index):
                end = (end[0] - 5, end[1])
            else:
                end = (end[0] + 5, end[1])

            while gameState.hasWall(end[0], end[1]):
                end = random.choice(self.entryPoints)

                if gameState.isOnRedTeam(self.index):
                    end = (end[0] - 5, end[1])
                else:
                    end = (end[0] + 5, end[1])

        avoid = set()
        if gameState.isOnRedTeam(self.index):
            for pos in self.entryPoints:
                avoid.add((pos[0] + 1, pos[1]))
        else:
            for pos in self.entryPoints:
                avoid.add((pos[0] - 1, pos[1]))

        path = bfs(self.index, gameState, self, end, avoid)

        if len(path) > 0:
            pos = path.pop(0)
            pos = (int(pos[0]), int(pos[1]))
            # self.debugDraw(path, (1.0, 1.0, 1.0), True)
            self.otherPath = True
            self.path = path
            if (pos[0] - myPos[0], pos[1] - myPos[1]) in invDirections.keys():
                return invDirections[(pos[0] - myPos[0], pos[1] - myPos[1])]
    if len(path) > 0 and self.otherPath:
        pos = path.pop(0)
        self.path = path
        pos = (int(pos[0]), int(pos[1]))
        if (pos[0] - myPos[0], pos[1] - myPos[1]) in invDirections.keys():
            return invDirections[(pos[0] - myPos[0], pos[1] - myPos[1])]

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)

    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(gameState)]
    foodList = self.getFood(successor).asList()

    if len(foodList) > 2:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      invaderDistance = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders])
      features['invaderDistance'] = invaderDistance

    if successor.getAgentState(self.index).isPacman and self.defense:
      features['donotcross'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'invaderDistance': -10, 'donotcross': -500, 'distanceToFood': -1}


def bfs(index, gameState, agentState, end, avoid):
    start = gameState.getAgentPosition(index)
    if end == None:
        end = agentState.entryPoints[-1]

        m = float('inf')
        for e in agentState.entryPoints:
            if agentState.getMazeDistance(start, e) < m:
                m = agentState.getMazeDistance(start, e)
                end = e

    # Start and goal state obtained at this point
    from util import Queue
    fringe = Queue()
    current = start
    explored = set()
    explored.add(current)
    parent = {start: ("", "")}

    directions = {
        'North': (0, -1),
        'West': (-1, 0),
        'East': (1, 0),
        'South': (0, 1)
    }

    while not current == end:
        actions = getLegalActions(gameState, current)

        for action in actions:
            pos = (current[0] + directions[action][0], current[1] + directions[action][1])
            if pos not in explored and pos not in avoid:
                fringe.push(pos)
                parent[pos] = current
                explored.add(pos)

        if fringe.isEmpty(): break
        current = fringe.pop()

    if fringe.isEmpty():
        return []

    path = []
    current = end
    # reconstruct solution
    while parent[current] != ("", ""):
        path.append(current)
        current = parent[current]
    path = list(reversed(path))

    return path

def getLegalActions(gameState, position):
    directions = {
        'North': (0, -1),
        'West': (-1, 0),
        'East': (1, 0),
        'South': (0, 1)
    }
    actions = []
    for dir in directions.keys():
        newPos = (position[0] + directions[dir][0], position[1] + directions[dir][1])
        if not gameState.hasWall(newPos[0], newPos[1]):
            actions.append(dir)

    return actions