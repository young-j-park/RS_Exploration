
import numpy as np


class Evaluator:
    def __init__(self, env):
        self.exposed = []
        self.responses = []
        self.num_total_exposed = 0
        self.num_total_responses = 0
        documents = env.env.candidate_set.get_all_documents()
        self.num_documents = len(documents)
        self.document_topics = {i: documents[i].cluster_id for i in range(self.num_documents)}
        self.document_qualities = {i: documents[i].quality for i in range(self.num_documents)}
        self.num_users = len(env.env.user_model)
        self.user_topic_affinity = np.array(
            [env.env.user_model[i].get_topic_affinity() for i in range(self.num_users)]
        )
        user_topic_sorted = np.argsort(-self.user_topic_affinity, axis=1)
        self.user_topic_ranks = [
            dict(zip(user_topic_sorted[i], range(1, self.num_documents+1)))
            for i in range(self.num_users)
            ]

    def add(self, slates, responses):
        self.exposed.append(slates)
        self.responses.append(responses)

        self.num_total_responses += np.sum(responses)
        self.num_total_exposed += slates.size

    def hit_ratio(self):
        return self.num_total_responses / self.num_total_exposed

    def ranking(self):
        exposed_topics = np.vectorize(self.document_topics.get)(self.exposed)
        mrrs_all = []
        for topics in exposed_topics:
            mrrs = []
            for i_user in range(self.num_users):
                mrr = 0
                for doc in topics[i_user]:
                    mrr += 1/self.user_topic_ranks[i_user][doc]
                mrrs.append(mrr)
            mrrs_all.append(mrrs)
        return mrrs_all
