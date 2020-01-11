from QAgent import QAgent
from SAgent import SAgent

if __name__ == "__main__":
    agents = [('QAgent', QAgent()), ('SAgent', SAgent())]
    result = {n: [] for n, a in agents}

    for name, agent in agents:
        agent.train()
        for i in range(100):
            print(f'Simulation {i} for: {name}')
            result[name].append(agent.simulate())
        agent.shutdown()

    for k, v in result.items():
        v = [i for i in v if not i]
        print(k, len(v))
