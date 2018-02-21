import gym
import ailib
import numpy as np

#env = gym.make("Acrobot-v1")
env = gym.make('CartPole-v1')

agent_model = ailib.neuralnetwork()
agent_model.push_variable("INPUT", env.observation_space.shape[0])
agent_model.push_linear("FC1", "INPUT", 32)
agent_model.push_normalization("NOR1", "FC1", 0.0)
agent_model.push_relu("ACT1", "NOR1")
agent_model.push_linear("FC2", "ACT1", 32)
agent_model.push_normalization("NOR2", "FC2", 0.0)
agent_model.push_relu("ACT2", "NOR2")
agent_model.push_linear("FC3", "ACT2", env.action_space.n)

agent_model.printstack()

agent = ailib.dqn_agent(agent_model, 200000, 0.003, 0.9, 0.97)

for i_episode in range(1000):
    observation = env.reset()
    agent.set_curiosity(agent.get_curiosity() * 0.99)
    for t in range(500):
        if i_episode % 10 == 0: env.render()
        #env.render()
        action = agent.act(observation.astype("f"))
        observation, reward, done, info = env.step(action)
        #print(agent_model.getoutput("FC3"))
        if done: agent.observe(np.array([]).astype("f"), reward)
        else: agent.observe(observation.astype("f"), reward)
        if done:
            print("Episode finished after {} timesteps, lr: {} curiosity: {} lte: {}".format(t+1,
                agent.get_learningrate(), agent.get_curiosity(), agent.get_longtermexploitation()))
            break
