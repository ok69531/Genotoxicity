class TG473Parser():
    def __init__(self):
        super().__init__()
        
        self.btach_size = 128
        self.num_layers = 4
        self.hidden = 1024
        self.epochs = 300
        self.lr = 0.003
        self.optimizer = 'sgd'
        self.weight_decay = 1e-4


tg473_args = TG473Parser()
