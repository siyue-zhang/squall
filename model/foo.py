# import torch                                                                                                                                                  
# import torch.nn as nn                                                                                                                                         
                                                                                                                                                              
# from torch.nn.utils.rnn import pack_padded_sequence                                                                                  

# cudnn_version = torch.backends.cudnn.version()
# print(f"cuDNN version: {cudnn_version}")

# cuda_version = torch.version.cuda
# print(f"CUDA Version: {cuda_version}")

# torch.manual_seed(1)                                                                                                                                          
                                                                                                                                                              
# lstm = nn.LSTM(3, 3).cuda()                                                                                                                                   
# inputs = torch.randn((3,3,3)).to(device='cuda')                                                                                                               
                                                                                                                                                              
                                                                                                                                                              
# packed = pack_padded_sequence(inputs, [3,3,3])                                                                                                                
# out, hidden = lstm(packed)                                                                                                                                    
                                                                                                                                                              
# print(out)                                                                                                                                                    
# print(hidden)

import logging
from evaluator import Evaluator

evaluator = Evaluator(
        "../tables/tagged/",
        "../tables/db/",
        "../stanford-corenlp-full-2018-10-05/"
)

prediction = [
    {'table_id': '203_447', 
     'result': [{'sql': "select c5_number from w where c3 = 'antonio horvath kiss'",
                 'sql_type': '',
                #  'sql_type': 'Keyword Column Keyword Keyword Keyword Column Keyword Literal.String Keyword', 
                 'id': 'nt-12427', 
                 'tgt': "select c3 from w where c3_first = 'antonio horvath kiss'", 
                 'nl': 'when was antonio horvath kiss last elected ?'
                 }]
    }]
predictions = [prediction]

lf_accu = 0
all_accu = 0
total = 0
all_preds = list()
log_probs = []


for idx, pre in enumerate(predictions):
    ex_acc = evaluator.evaluate(prediction)
    print("ex_acc: ", ex_acc)
    if idx == 1:
        print("example prediction: ")
        print(prediction)
    for d in prediction:
        total += 1
        if d['result'][0]['sql'] == d['result'][0]['tgt']:
            lf_accu += 1
    if ex_acc == 1:
        prediction[0]['correct'] = 1
    else:
        prediction[0]['correct'] = 0
    all_preds.extend(prediction)
    all_accu += ex_acc

print('logical form accurate: {}/{} = {}%'.format(lf_accu, total, lf_accu / total * 100))
print('num of execution correct: {}/{} = {}%'.format(all_accu, total, all_accu / total * 100))

