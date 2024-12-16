class GIBParser():
    def __init__(self, inner_loop = None, beta = None, pp_weight = None):
        super().__init__()
        
        self.inner_loop = inner_loop
        self.beta = beta
        self.pp_weight = pp_weight


gib_args = GIBParser()
