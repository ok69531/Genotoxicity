class TG474Parser():
    def __init__(self):
        super().__init__()
        
        self.batch_size = 128
        self.num_layers = 6
        self.hidden_dim = 1024
        self.epochs = 100
        self.lr = 0.001
        self.optimizer = 'sgd'
        self.weight_decay = 1e-5


tg474_args = TG474Parser()
