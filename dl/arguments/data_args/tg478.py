class TG478Parser():
    def __init__(self):
        super().__init__()
        
        self.btach_size = 64
        self.num_layers = 2
        self.hidden = 64
        self.epochs = 100
        self.lr = 0.001
        self.optimizer = 'adam'
        self.weight_decay = 1e-4


tg478_args = TG478Parser()
