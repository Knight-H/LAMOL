import os, json, logging
import itertools
from rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl, Annotation
import numpy as np
from scipy import stats
from pathlib import Path 
import torch
from pytorch_transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel #, GPT2Model, 
from typing import Any, Callable, Dict, List, Set, Tuple
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

model_name = 'gpt2'

OLD_MODEL_DIR = Path("./task1_movie")
OLD_TOK_DIR = Path("./task1_movie")

NEW_MODEL_DIR = Path("./task2_scifact")
NEW_TOK_DIR = Path("./task2_scifact")

device = torch.device('cuda:0')
# device = torch.device('cpu')

logging.basicConfig(filename='output.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARN)


## Import Old Model 
model_old_config = GPT2Config.from_json_file(OLD_MODEL_DIR/"config.json")
model_old_config.output_attentions = True
model_old = GPT2LMHeadModel(model_old_config)
model_old.load_state_dict(torch.load(OLD_MODEL_DIR/"model-5"))

## Import New Model 
model_new_config = GPT2Config.from_json_file(NEW_MODEL_DIR/"config.json")
model_new_config.output_attentions = True
model_new = GPT2LMHeadModel(model_new_config)
model_new.load_state_dict(torch.load(NEW_MODEL_DIR/"model-5"))

model_old.to(device)
model_new.to(device)


# From LAMOL/settings.py
# special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
# tokenizer.add_tokens(list(special_tokens.values()))

with open(NEW_TOK_DIR/"special_tokens_map.json") as f:
    special_tokens_map = json.load(f)
print(f"special_tokens_map: {special_tokens_map}")

with open(NEW_TOK_DIR/"added_tokens.json") as f:
    added_tokens = json.load(f)
print(f"added_tokens: {added_tokens}")


tokenizer = GPT2Tokenizer(NEW_TOK_DIR/"vocab.json", NEW_TOK_DIR/"merges.txt")
tokenizer.add_tokens(list(added_tokens.keys()))
# print(token)
print(f"Total # of tokens: {len(tokenizer)}")


tokenizer.ans_token = "__ans__"
tokenizer.ans_token_id = tokenizer.convert_tokens_to_ids("__ans__")


for k,v in special_tokens_map.items():
    assert tokenizer.special_tokens_map[k] == v
for tok, tok_id in added_tokens.items():
    assert tokenizer.convert_ids_to_tokens(tok_id) == tok
print("<special_tokens_map and added_tokens matched successfully>")


from rationale_benchmark.utils import (
    annotations_from_jsonl,
    load_flattened_documents
)
from itertools import chain

data_root = os.path.join('data', 'movies')
annotations = annotations_from_jsonl(os.path.join(data_root, 'val.jsonl'))

docids = sorted(set(chain.from_iterable((ev.docid for ev in chain.from_iterable(ann.evidences)) for ann in annotations)))
flattened_documents = load_flattened_documents(data_root, docids)


key_to_annotation = dict()
for ann in annotations:
    # For every evidence in the evidence list of the annotation, 
    # find the document id as the key, 
    # mark True for every start_token and end_token
    for ev in chain.from_iterable(ann.evidences):
#         key = (ann.annotation_id, ev.docid) # THIS IS THE SAME, not sure why they make it a tuple, maybe for defensive programming?
        if (ann.annotation_id != ev.docid):
            raise Exception("Annotation ID and Doc ID must be the same!!!")
        key = ann.annotation_id
    
        if key not in key_to_annotation:
            key_to_annotation[key] = [False for _ in flattened_documents[ev.docid]]
        
        start, end = ev.start_token, ev.end_token
        
        for t in range(start, end):
            key_to_annotation[key][t] = True
            
annotation_dict = dict()
for ann in annotations:
    # From key annotation_id/docid -> annotation itself
    annotation_dict[ann.annotation_id] = ann
    
    
def _auprc(true, pred):
    true = [int(t) for t in true]
    precision, recall, _ = precision_recall_curve(true, pred)
    return auc(recall, precision)

def _avg_auprc(truths, preds):
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    aucs = []
    for k, true in truth.items():
        pred = preds[k]
        aucs.append(_auprc(true, pred))
    return np.average(aucs)

def convert_to_model_input(document, question, answer, tokenizer, modelConfig, device, return_tensors=True):
    """Input is a string of the document, question, and answer
    Refer to https://github.com/jojotenya/LAMOL/blob/03c31d9f0c7bf71295bc2d362ddf40a7656956e1/utils.py#L220
    
    Outputs:
        context[:args.max_len-len(example)-1] + question + ans_token + answer
        maximum of 1023 length, since the original -1 for the eos_token at the end
    """
    # Need to manually truncate it to 1024 [GPT2]
    if isinstance(document, list): # Pretokenized input, just need to convert it to tokens.
        document = tokenizer.convert_tokens_to_ids(document)
    elif isinstance(document, str): # Tokenize and encode it
        document = tokenizer.encode(document)
    else:
        raise Exception("Document should be list or string")
    question = tokenizer.encode(question)
    answer   = tokenizer.encode(answer)
    
    example = question + [tokenizer.ans_token_id] + answer
    
    if len(example) + 1 > modelConfig.n_ctx:
        logger.warning('an example with len {} is too long!'.format(len(example) + 1))
        return
    
    # -1 because there is eos_token spare for the original LAMOL
    _input = document[:modelConfig.n_ctx-len(example)-1] + example
    
    document_mask = np.zeros((len(_input)), dtype=bool)
    document_mask[:len(document[:modelConfig.n_ctx-len(example)-1])] = True
    
    # Convert to Tensors if required
    if return_tensors:
        _input = torch.tensor(_input, dtype=torch.long, device=device)
        
    return {
        'input_ids': _input,
        'document_mask': document_mask,
    }

def convert_to_tokenized_ground_truth(original_ground_truth, original_document, tokenizer):
    """ Algorithm to get new_ground_truth by the tokenizer. Checking each substring if it's equal, and appending the 
    ground_truth value of the original_document_index
    Assumptions: NO UNKNOWNS! since we check by ==, else need to check for unknowns and perform equality ignoring left side.
    
    Inputs:
        original_ground_truth: Original GT boolean array with same shape as original_document
        original_document: Original Pretokenized document array with same shape as original_ground_truth
        tokenizer: tokenizer used to encode/decode the document
        
    Output: 
        new_ground_truth: New GT boolean array expanded by tokenizer
    """
    new_document = tokenizer.encode(' '.join(original_document))
    new_ground_truth  = []
    
    original_document_start_index = 0
    original_document_end_index = 1
    new_document_start_index = 0
    new_document_end_index = 1
    
    while new_document_end_index <= len(new_document):
        original_document_temp = ' '.join(original_document[original_document_start_index:original_document_end_index])
        new_document_temp = tokenizer.decode(new_document[new_document_start_index:new_document_end_index]).strip()
        
        new_ground_truth.append(original_ground_truth[original_document_end_index-1])
        
#         if new_document_end_index < 150:
#             print("NEW DOC", new_document_temp)
#             print("ORI DOC", original_document_temp)
#             print(new_ground_truth)
        
        ## ASSUME THAT NEW_DOCUMENT_TEMP HAS NO UNKNOWNS??!?
        if new_document_temp == original_document_temp:
            original_document_start_index += 1
            original_document_end_index += 1
            new_document_start_index = new_document_end_index
        
        new_document_end_index += 1
        
    
    return new_ground_truth

def select_attention(single_attention_head):
    """Returns the aggregated results of all the tokens
    Currently just use CLS"""
#     return attention_head[0]
    # Try Averaging
    return single_attention_head.mean(axis=0)

def find_attn_head_max(attention_block, ground_truth, mask, method="auprc"):
    """Input 
        attention block (with attention heads): Dimension  [attention_head, seq_len, seq_len]
        ground_truth/feature map              : Dimension  [seq_len]
        mask                                  : Dimension  [seq_len] 
        method                                : "auprc"/"iou"/"auprc-token-level"

    Returns 
        attn_head_max_index          : attention head index of the max auprc 
        auprc_max                    : the value of the max auprc
        attn_head_token_max_index    : attention head token index (for token-granularity only)
    """
    if device.type == "cuda":
        attention_block = attention_block.cpu().detach()[:, :mask.sum(), :mask.sum()]
    else:
        attention_block = attention_block.detach()[:, :mask.sum(), :mask.sum()]
    ground_truth    = ground_truth[:mask.sum()]  # Since ground_truth has undefined length, may be higher
    logger.debug(f"ATTN BLOCK SHAPE {attention_block.shape}")

    # auprc default is the attention_head level, aggregated by select_attention
    if method=="auprc":
        auprcs = []
        for attn_head in attention_block:
            pred = select_attention(attn_head)

            auprc = _auprc(ground_truth,pred)
            auprcs.append(auprc)

        attn_head_max_index = np.argmax(auprcs)
        return attn_head_max_index, auprcs[attn_head_max_index]
    
    # auprc-token-level is the token level, not aggregated. for loop another level!
    elif method=="auprc-token-level":
        auprcs = []
        
        # Attn_head is Dimension [seq_len, seq_len]
        for attn_head_ind, attn_head in enumerate(attention_block):
            
            logger.debug(f"atten head {attn_head_ind} {attn_head.shape}")
            # Attn_head_token is Dimension [seq_len], for each token compared to other tokens
            for attn_head_token in attn_head:
                pred = attn_head_token
#                 logger.debug(f"atten head token {attn_head_token.shape} ")
#                 logger.debug(f"ground truth {len(ground_truth)} ")
                auprc = _auprc(ground_truth,pred)
                auprcs.append(auprc)
        
        attn_head_token_max_index = np.argmax(auprcs)
        attn_head_max_index = attn_head_token_max_index // attention_block.shape[-1] # Divided by seq len to get the max attention_head 
        token_max_index     = attn_head_token_max_index % attention_block.shape[-1]  #Remainder of seq len to get token index
        logger.info(f"LEN auprc: {len(auprcs)} Argmax of AUPRC: {np.argmax(auprcs)} MAX auprc: {auprcs[attn_head_token_max_index]}")
        logger.info(f"attn_head_max_index: {attn_head_max_index} auprcs:10: {auprcs[:10]}")
        return attention_block[attn_head_max_index][token_max_index], auprcs[attn_head_token_max_index]
            
    elif method=="iou":
        ious = []
#         ground_truth = ground_trou
        for attn_head in attention_block:
            pred = select_attention(attn_head).detach()

    #         print(len(ground_truth))
    #         print(pred.shape)

            auprc = _auprc(ground_truth,pred)
            auprcs.append(auprc)
    #         print(auprc)
    #         print(pred)

#         print(auprcs)
        attn_head_max_index = np.argmax(auprcs)
#         print(attn_head_max_index, auprcs[attn_head_max_index])
        return attn_head_max_index, auprcs[attn_head_max_index]

MO_GT_METHOD = "auprc-token-level"
MN_MO_METHOD = "auprc-token-level"

block_L = []

for docid in tqdm(docids):
    
    logger.info(f"Document ID: {docid}")
    logger.info(f"Document: {' '.join(flattened_documents[docid][:20])}")
    logger.info(f"Question: {annotation_dict[docid].query}")
    logger.info(f"Answer: {annotation_dict[docid].classification}")
    
    ### 1. Convert Document, Question, Answer to model input ###
    # Note: if send in pretokenized document, only convert it to ids, but for query and classification needs to tokenize
    #       if send in string, then will tokenize but ground_truth needs to convert_to_tokenized_ground_truth!!
    _input = convert_to_model_input(' '.join(flattened_documents[docid]), 
                                    annotation_dict[docid].query,  
                                    annotation_dict[docid].classification,
                                   tokenizer,
                                   model_new_config,
                                   device)
    
    input_ids = _input['input_ids']
    document_mask = _input['document_mask']
    ground_truth = convert_to_tokenized_ground_truth(key_to_annotation[docid], flattened_documents[docid], tokenizer)
    
    input_ids = input_ids.reshape([1, -1])
    
    logger.info(f"Input Shape: {input_ids.shape}")
    logger.debug(tokenizer.decode(input_ids.squeeze().tolist()))
    logger.info(f"Document Mask Sum: {document_mask.sum()}")

    ### 2. Predict the attentions from the input tokens ###
    last_hidden_state_old, pooler_output_old, attentions_old = model_old(input_ids)
    logger.info(f"Attention Blocks: {len(attentions_old)} First attention block old shape: {attentions_old[0].shape}")
    
    last_hidden_state_new, pooler_output_new, attentions_new = model_new(input_ids)
    logger.info(f"Attention Blocks: {len(attentions_new)} First attention block new shape: {attentions_new[0].shape}")
    

    block_auprcs = [] # List of max IOU MO-MN
    block_rm = []     # List of representative maps of MO-MN
    
    
    # attentions is a list of attention blocks (12), 
    #   where each attention has the dimension [batch_size, attention_head, seq_len, seq_len]
    for block_index, [block_old, block_new] in enumerate(zip(attentions_old, attentions_new)):
    
        # block first dimension is batchsize! - need to squeeze it out since it's always (1)
        # Block has dimension [batch_size, attention_head, seq_len, seq_len] where batch_size=1
        block_old = block_old.squeeze() # Dimension  [attention_head, seq_len, seq_len]
        block_new = block_new.squeeze() # Dimension [attention_head, seq_len, seq_len]
        logger.debug(f"Block Old Shape: {block_old.shape}")
        logger.debug(f"Block New Shape: {block_new.shape}")
        
        rm_mo_gt, max_mo_gt = find_attn_head_max(block_old, ground_truth, document_mask, method=MO_GT_METHOD)
        
        # Change rm_mo_gt Representative Map of Old model and Ground Truth -> Boolean Array for top 20 percentile
        rm_mo_gt_top20 = rm_mo_gt > np.percentile(rm_mo_gt, 80)
        rm_mn_mo, max_mn_mo = find_attn_head_max(block_new, rm_mo_gt_top20, document_mask, method=MN_MO_METHOD)
        
        
        # Append the maximum AuPRC 
        block_auprcs.append(max_mn_mo)
        # Append the RM MN MO
        block_rm.append(rm_mn_mo)

    
    # Block with highest drop in IOU
    b = np.argmin(block_auprcs)
    block_L.append(b)
    print(block_L)

# Most frequent block in block_L
print("Most frequent block:" ,stats.mode(block_L))

cnt = Counter()
for block in block_L:
    cnt[block] += 1
print(cnt)