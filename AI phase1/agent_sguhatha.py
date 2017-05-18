from agents import Agent

class Agent_sguhatha(Agent):
    def will_buy(self, value, price, prob):
        return (price/value<=0.8 and prob>=0.4)
    
