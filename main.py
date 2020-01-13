from QAgent import QAgent
from SAgent2 import SAgent2
from SAgent import SAgent

if __name__ == "__main__":
    agents = [('QAgent', QAgent()), ('SAgent_RBF', SAgent()), ('SAgent_Discretisize', SAgent2())]

    result = {n: [] for n, a in agents}

    for name, agent in agents:
        agent.train()
        for i in range(1000):
            print(f'Simulation {i} for: {name}')
            result[name].append(agent.simulate(render=False))
        agent.shutdown()

    for k, v in result.items():
        v = [i for i in v if not i]
        print(k, len(v))
