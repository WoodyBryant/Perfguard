import torch
from perfguard import PerfGuard
from get_data import *
import collections
import json
import ImportConfig
config = ImportConfig.Config()

def test():

    # merge pg_plan and lero_plan(or other plan chosen by the AI model)
    plan_dict = collections.defaultdict(list)
    with open(config.pg_plan_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(config.SEP)
            plan_dict[arr[0]].append(json.loads(arr[1]))

    with open(config.lero_plan_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(config.SEP)
            if arr[0] in plan_dict.keys():
                plan_dict[arr[0]].append(json.loads(arr[1]))
            
    # preprocess testing data
    qids  = []
    for qid in plan_dict.keys():
        qids.append(qid)
    
    get_data_ = Get_Dataset_Test(plan_dict)
    model = get_data_.load_model(config.model_name,False)
    features1,features2 = get_data_.get_features()
    adjaceny_matrix_list_x1,adjaceny_matrix_list_x2 = get_data_.get_two_adjaceny_matrix()
    predict = model(adjaceny_matrix_list_x1,adjaceny_matrix_list_x2,features1,features2)

    # test
    predict = predict.cpu().detach().numpy().tolist()
    predict_label= [1 if x>config.threshold else 0 for x in predict]
    label_dict = dict(zip(qids,predict_label))
    latency_list = []
    perfguard_dict = {}
    for qid in label_dict.keys():
        if len(plan_dict[qid]) == 2:
            if label_dict[qid] == 1:
                perfguard_dict[qid] = plan_dict[qid][0]
            else:
                perfguard_dict[qid] = plan_dict[qid][1]
            latency_list.append(perfguard_dict[qid][0]['Execution Time']/1000)

    # save results
    with open(config.perfguard_path,'w') as f:
        for qid in perfguard_dict.keys():
            f.write(qid+config.SEP+json.dumps(perfguard_dict[qid])+'\n')
    print(sum(latency_list))
    

if __name__ == '__main__':
    test()

