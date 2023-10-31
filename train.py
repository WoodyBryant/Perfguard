import torch
from perfguard import PerfGuard
from get_data import *
import ImportConfig
config = ImportConfig.Config()

def train():

    # preprocess training data
    get_data_ = Get_Dataset(config.train_file)
    features1,features2 = get_data_.get_features()
    label = get_data_.get_labels()
    adjaceny_matrix_list_x1,adjaceny_matrix_list_x2 = get_data_.get_two_adjaceny_matrix()

    # model training
    model = PerfGuard(features1.shape[2],config.embd_dim, config.tensor_dim,config.dropout).cuda(config.device)
    model= torch.nn.DataParallel(model, device_ids=config.GPU_LIST)
    optimizer = torch.optim.Adam(model.parameters(), config.init_lr)
    Loss = torch.nn.BCELoss()
    model.train()
    for epoch in range(config.epochs):
        final_output = model(adjaceny_matrix_list_x1,adjaceny_matrix_list_x2,features1,features2)
        
        # input dim of gcn : M*D and M*M
        # output dim of gcs : M*F
        loss = Loss(final_output, torch.tensor(label).float().cuda(config.device))
        print("Epoch {}, loss {}".format(epoch+1,loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # model saving
    get_data_.save(model,config.model_name,features1.shape[2])

if __name__ == '__main__':
    train()




 



