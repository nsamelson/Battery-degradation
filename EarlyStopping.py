import jax


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_params = None  # To store the best parameters

    def __call__(self, current_loss, params):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)  # Deep copy
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
