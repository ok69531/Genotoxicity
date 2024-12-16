class VGIBParser():
    def __init__(self, mi_weight = None, con_weight = None):
        super().__init__()
        
        self.second_dense_neurons = 2
        
        self.mi_weight = mi_weight
        self.con_weight = con_weight


vgib_args = VGIBParser()
