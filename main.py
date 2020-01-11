from QAgent import QAgent
from SAgent import SAgent

if __name__ == "__main__":
    for agent in [SAgent(), QAgent()]:
        agent.train()

        for i in range(10):
            agent.simulate()
        agent.shutdown()
