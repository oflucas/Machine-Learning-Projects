import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    ALFA = 0.5     # Learning Rate
    GAMMA = 0.5     # Discount Factor

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.EPSON = 0.1     # Explore Probability
        self.nTrials = 100
        self.kEpson = -0.01 # Epson dcrease slope
        self.qInit = 2 # Q table initial values

        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.Q = {}
        self.t = 0          # Trial times count
        self.track_epson = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.t += 1
        self.track_epson.append(100 * min(1, max(0, self.EPSON + self.kEpson * self.t)))

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state && state defination
        self.state = (inputs['light'], self.next_waypoint)

        # TODO: Learn policy based on current & previous state, previous action, reward got during state transition
        self.learnDrive()
        
        # TODO: Select action according to your policy and current state
        self.action = self.takeAction()

        # Execute action and get reward
        self.reward = self.env.act(self, self.action)
        self.lastState = self.state
        self.lastAction = self.action
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, self.reward)  # [debug]

    def learnDrive(self):
        ### Q(lastState,lastAction) <-Alfa- reward + Gamma * max{Q(currentState,AllActions)}
        lastQ = self.queryQ((self.lastState, self.lastAction))
        
        # find maxQ for all actions for current state
        maxQ = 0
        for i_action in Environment.valid_actions:
            maxQ = max(maxQ, self.queryQ((self.state, i_action)))

        # update Q
        self.Q[(self.lastState, self.lastAction)] = lastQ * (1 - LearningAgent.ALFA) + LearningAgent.ALFA * (self.reward + LearningAgent.GAMMA * maxQ)

    def queryQ(self, state_action_pair):
        if state_action_pair == None: return self.qInit
        return self.Q.setdefault(state_action_pair, self.qInit) # if not contain the key, init it with self.qInit

    def takeAction(self):
        epsilon = self.EPSON + self.kEpson * self.t
        draw = random.random()
        #print "draw = ", draw
        if draw < epsilon:
            #print "choose to explore"
            return random.choice(Environment.valid_actions)
        else:
            #print "choose to use learned"
            maxQ = -9999
            maxAction = None
            for i_action in Environment.valid_actions:
                curQ = self.queryQ((self.state, i_action))
                if maxQ < curQ :
                    maxQ = curQ
                    maxAction = i_action
            return maxAction


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    #Visualize the train result
    # sim = Simulator(e, update_delay=1, display=True)
    # sim.run(n_trials=3)

    # Analysis
    # print "DEADLINE TRACK: \n", e.track_deadline
    # print "NET REWARD TRACK: \n", e.track_net_reward

    # import matplotlib.pyplot as plt
    # plt.plot(e.track_deadline, label = "Deadline")
    # plt.plot(e.track_net_reward, label = "Net reward")
    # plt.plot(a.track_epson, label = "Epsilon [%]")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run()
