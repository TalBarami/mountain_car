from continuous.ActorCriticAgent import ActorCriticAgent
from continuous.ActionDiscretizeAgent import ActionDiscretizeAgent
from discrete.QTableAgent import QTableAgent
from discrete.DiscretizeAgent import DiscretizeAgent
from discrete.RBFAgent import RBFAgent

if __name__ == "__main__":
    agents = [ActorCriticAgent()]

    result = {a.name: [] for a in agents}

    for agent in agents:
        agent.train()
        for i in range(100):
            print(f'Simulation {i} for: {agent.name}')
            result[agent.name].append(agent.simulate(render=False))
        agent.shutdown()

    for k, v in result.items():
        v = [i for i in v if not i]
        print(k, len(v))
