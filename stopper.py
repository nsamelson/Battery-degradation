from ray.tune import Stopper

class GlobalNoImprovementStopper(Stopper):
    def __init__(self, patience=50, metric="cae"):
        self.best_score = None
        self.counter = 0
        self.patience = patience
        self.metric = metric

    def __call__(self, trial_id, result):
        current_score = result[self.metric]
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        return False  # We never stop individual trials

    def stop_all(self):
        return self.counter >= self.patience
