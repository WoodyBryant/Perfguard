import torch
class Config():
    def __init__(self):
        # cuda
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu")
        self.GPU_LIST = [0]
        self.SEP = "#####"

        # hyper parameter
        self.init_lr = 0.1
        self.epochs = 500
        #the output dim of gcn
        self.embd_dim = 30
        #the out put dim of ntn
        self.tensor_dim = 10
        self.dropout = 0.1
        self.threshold = 0.5

        # train
        self.model_name = "model_pth/tmp"
        self.train_file = "data/lero_tpch4.log.training"

        # test
        self.lero_plan_path = "data/lero_tpch.log.testing"
        self.pg_plan_path = "data/pg_tpch.log.tesing"
        self.perfguard_path = "data/perfguard_tpch.log.testing"

