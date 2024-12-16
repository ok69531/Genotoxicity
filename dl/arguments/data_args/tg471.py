class TG471Parser():
    def __init__(self):
        super().__init__()
        
        self.batch_size = 64
        self.num_layers = 2
        self.hidden_dim = 512
        self.epochs = 300
        self.lr = 0.003
        self.optimizer = 'sgd'
        self.weight_decay = 0


tg471_args = TG471Parser()
