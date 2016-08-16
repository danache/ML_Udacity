import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    successnum = []
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Qtabel = dict()
        self.alpha = 0.15
        self.gamma = 0.6
        self.epsilon = 0.1

        self.lastState = 0
        self.lasAction = 0
        self.lastReward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.lastState = 0
        self.lasAction = 0
        self.lastReward = 0


    def setArgument(self, argument):
        self.alpha = argument[0]
        self.gamma = argument[1]
        self.epsilon = argument[2]
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'] ,inputs['oncoming'] ,inputs['right'] ,inputs['left'])
        tmp = []

        for i in range(0, 4):

            tup = (self.state, self.env.valid_actions[i])

            if tup not in self.Qtabel:
                self.Qtabel[tup] = 0

            tmp.append(self.Qtabel[tup])

        maxaction = max(tmp)
        if (self.lastState, self.lasAction) not in self.Qtabel:
            self.Qtabel[(self.lastState, self.lasAction)] = 0
        self.Qtabel[(self.lastState, self.lasAction)] = ((1-self.alpha) * self.Qtabel[(self.lastState, self.lasAction)] \
                                                        + self.alpha * (self.lastReward + self.gamma * maxaction))
        # TODO: Select action according to your policy
        action = None
        res= []
        for i in range(0,4):
            if(self.Qtabel[(self.state, self.env.valid_actions[i])] == maxaction):
                res.append(i)
        a = random.random()
        if self.epsilon < a:
            if len(res) == 1:
                action = self.env.valid_actions[res[0]]
            elif len(res) > 1:
                ran = random.choice(res)
                action = self.env.valid_actions[ran]
        else:
            action = self.env.valid_actions[random.randint(0,3)]



        # Execute action and get reward
        reward = self.env.act(self, action)
        self.lastState = self.state
        self.lasAction = action
        self.lastReward = reward
        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print(self.state)
def run():
    """Run the agent for a finite number of trials."""
    successnum = dict()
    for i in range(10, 36,10):
        for j in range(40,71,10):
            for k in range(6,16,4):
                arguemns = (i/100.0, j/100.0, k/100.0)
                tenSucc = []
                for index in range(0, 5):
                    # Set up environment and agent
                    e = Environment()  # create environment (also adds some dummy traffic)
                    a = e.create_agent(LearningAgent,arguemns)  # create agent
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                    # Now simulate it
                    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
                    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                    sim.run(n_trials=100)  # run for a specified number of trials
                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    tenSucc.append(e.success)
                successnum[arguemns] = tenSucc

    print(successnum)

if __name__ == '__main__':
    run()
