class TG487Parser():
    def __init__(self):
        super().__init__()
        
        self.btach_size = 64
        self.num_layers = 6
        self.hidden = 64
        self.epochs = 300
        self.lr = 0.001
        self.optimizer = 'adam'
        self.weight_decay = 1e-5


tg487_args = TG487Parser()
