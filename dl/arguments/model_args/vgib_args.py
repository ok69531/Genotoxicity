class VGIBParser():
    def __init__(self):
        super().__init__()
        
        self.num_layers = 3
        self.epochs = 100
        self.hidden = 128
        self.second_dense_neurons = 2
        self.cls_hidden_dimensions = 64
        self.mi_weight = 10
        # self.mi_weight = 0.1
        self.con_weight = 3
        self.lr = 0.001
        self.weight_decay = 5*10**-5


vgib_args = VGIBParser()
