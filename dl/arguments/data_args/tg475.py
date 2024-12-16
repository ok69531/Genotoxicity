class TG475Parser():
    def __init__(self):
        super().__init__()
        
        self.batch_size = 128
        self.num_layers = 6
        self.hidden_dim = 128
        self.epochs = 300
        self.lr = 0.003
        self.optimizer = 'sgd'
        self.weight_decay = 1e-4


tg475_args = TG475Parser()
