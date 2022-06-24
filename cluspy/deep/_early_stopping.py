class EarlyStopping():
    """Early stopping to stop the training when the loss does not improve after
    certain epochs. Adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    
    Parameters
    ----------
    patience : int, default=10, how many epochs to wait before stopping when loss is not improving
    min_delta : float, default=1e-4, minimum difference between new loss and old loss for new loss to be considered as an improvement
    verbose : bool, default=False, if True will print INFO statements
    
    Attributes
    ----------
    counter : integer counting the consecutive epochs without improvement
    best_loss : best loss achieved before stopping
    early_stop : boolean indicating whether to stop training or not
    """

    def __init__(self, patience=10, min_delta=1e-4, verbose=False):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
