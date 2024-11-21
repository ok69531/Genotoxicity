class GSATParser():
    def __init__(self):
        super().__init__()
        
        self.epochs = 100
        self.batch_size = 32
        self.n_layers = 3
        self.hidden_size = 128
        self.dropout_p = 0.3
        self.use_edge_attr = True
        self.multi_label = False
        self.learn_edge_att = False
        self.extractor_dropout_p = 0.5
        self.lr = 0.001
        self.weight_decay = 0
        self.pred_loss_coef = 1.5
        # self.pred_loss_coef = 1
        self.info_loss_coef = 1
        self.fix_r = False
        self.decay_r = 0.1
        self.decay_interval = 10
        self.final_r = 0.5
        self.init_r = 0.9


gsat_args = GSATParser()
