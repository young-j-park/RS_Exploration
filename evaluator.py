
import numpy as np


class Evaluator:
    def __init__(self):
        self.exposed = []
        self.responses = []
        self.num_total_exposed = 0
        self.num_total_responses = 0

    def add(self, slates, responses):
        self.exposed.append(slates)
        self.responses.append(responses)

        self.num_total_responses += np.sum(responses)
        self.num_total_exposed += slates.size

    def hit_ratio(self):
        return self.num_total_responses / self.num_total_exposed
