import json
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# from source.utils.metrics import moses_multi_bleu, compute_prf
from misc import Pack

vocab_dic = {}
n_voc = open('../../preprocessing_scripts/kg/biobert/vocab/new_med_vocab.txt').read().split('\n')[:-1]

for i,item in enumerate(n_voc):
    vocab_dic[item] = i

vocab_size = len(vocab_dic)

print('vocab_size',vocab_size)

inv_voc_dic = {vocab_dic[item]: item for item in vocab_dic}

# def get_global_entity(entity_dir):
#     if entity_dir.endswith('KVR'):
#         entity_file = "%s/kvret_entities.json" % entity_dir
#     elif entity_dir.endswith('MULTIWOZ2.1'):
#         entity_file = "%s/global_entities.json" % entity_dir
#     elif entity_dir.endswith('CamRest'):
#         entity_file = "%s/camrest676-entities.json" % entity_dir
#     else:
#         print("Error when opening entity file!")
#         return
#     global_entity_list = []
#     with open(entity_file, 'r') as fr:
#         global_entity = json.load(fr)
#         for key in global_entity.keys():
#             if key != 'poi':
#                 global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
#             else:
#                 for item in global_entity['poi']:
#                     global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
#         global_entity_list = list(set(global_entity_list))
#     return global_entity_list    
     


def reward_fn(preds, targets):
    """
    Reward function
    """
    # weight parameters
    alpha1 = 1.0
    alpha2 = 1.0

    # acc reward
    '''
    # get the weighted mask
    no_padding_mask = preds.ne(self.padding_idx).float()
    trues = (preds == targets).float()
    if self.padding_idx is not None:
        weights = no_padding_mask
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    '''
    # print(preds.size())
    # print(targets.size())

    # print(preds)



    pred_text = [(' ').join([inv_voc_dic[i.item()] for i in item])  for item in preds.squeeze().tolist()]
    tgt_text = [(' ').join([inv_voc_dic[i.item()] for i in item])  for item in targets]
    print('pred_text',pred_text)
    batch_size = targets.size(0)

    result = Pack()
    result.add(pred_text=pred_text, tgt_text=tgt_text)
    result_list = result.flatten()

    # print(result_list)

    # bleu reward
    bleu_score = []
    for res in result_list:
        hyp_toks = res.pred_text.split()
        ref_toks = res.tgt_text.split()
        try:
            bleu_1 = sentence_bleu(references=[ref_toks], hypothesis=hyp_toks,
                                   smoothing_function=SmoothingFunction().method7,
                                   weights=[1, 0, 0, 0])
        except:
            bleu_1 = 0
        try:
            bleu_2 = sentence_bleu(references=[ref_toks], hypothesis=hyp_toks,
                                   smoothing_function=SmoothingFunction().method7,
                                   weights=[0.5, 0.5, 0, 0])
        except:
            bleu_2 = 0
        bleu = (bleu_1 + bleu_2) / 2
        bleu_score.append(bleu)
    bleu_score = torch.tensor(bleu_score, dtype=torch.float)

    # # entity f1 reward
    # f1_score = []
    # report_f1 = []
    # global_entity_list = get_global_entity(entity_dir)
    # for res in result_list:
    #     if len(res.gold_ents) == 0:
    #         f1_pred = 1.0
    #     else:
    #         gold_entity = res.gold_ents
    #         pred_sent = res.pred_text
    #         TP, FP, FN, f1_pred, _ = compute_prf(gold_entity, pred_sent,
    #                             global_entity_list=global_entity_list, 
    #                             kb_plain=res.kb_plain)
    #         report_f1.append(f1_pred)
    #     f1_score.append(f1_pred)
    # if len(report_f1) == 0:
    #     report_f1.append(0.0)
    # f1_score = torch.tensor(f1_score, dtype=torch.float)
    # report_f1 = torch.tensor(report_f1, dtype=torch.float)

    # if self.use_gpu:
    bleu_score = bleu_score.cuda()
        # f1_score = f1_score.cuda()
        # report_f1 = report_f1.cuda()

    # compound reward
    reward = alpha1 * bleu_score.unsqueeze(-1)# + alpha2 * f1_score.unsqueeze(-1)

    return reward, bleu_score#, report_f1