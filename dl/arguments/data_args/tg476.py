class TG476Parser():
    def __init__(self):
        super().__init__()
        
        self.btach_size = 32
        self.num_layers = 5
        self.hidden = 1024
        self.epochs = 100
        self.lr = 0.003
        self.optimizer = 'sgd'
        self.weight_decay = 1e-4


tg476_args = TG476Parser()
