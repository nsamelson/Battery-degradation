import jax


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, relative=True):
        self.patience = patience
        self.min_delta = min_delta  # Can be absolute or relative depending on `relative`
        self.relative = relative
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_params = None

    def __call__(self, current_loss, params):
        # Compute effective delta
        if self.relative and self.best_loss != float('inf'):
            effective_delta = self.best_loss * self.min_delta
        else:
            effective_delta = self.min_delta

        if current_loss < self.best_loss - effective_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
