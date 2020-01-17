from continuous.AgentQTableContinuous import AgentQTableContinuous
from discrete.AgentQTable import AgentQTable
from discrete.AgentDiscretize import AgentDiscretize
from discrete.AgentRBF import AgentRBF

if __name__ == "__main__":
    agents = [AgentQTable(), AgentDiscretize(), AgentRBF(), AgentQTableContinuous()]

    result = {a.name: [] for a in agents}

    for agent in agents:
        agent.train()
        for i in range(1000):
            print(f'Simulation {i} for: {agent.name}')
            result[agent.name].append(agent.simulate(render=False))
        agent.shutdown()

    for k, v in result.items():
        v = [i for i in v if not i]
        print(k, len(v))
