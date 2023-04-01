"""Implementation based on https://github.com/pcyin/pytorch_nmt and 
Stanford CS224 2019 class.

Usage:
    sanity_check.py 1d
    sanity_check.py 1e
    sanity_check.py 1f
"""
import math
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
from docopt import docopt
from tqdm import tqdm

from nmt_model import NMT
from utils import batch_iter, read_corpus
from vocab import Vocab, VocabEntry


BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)


def question_1d_sanity_check(model, src_sents, tgt_sents, vocab):
    """ Sanity check for question 1d. 
        Compares student output to that of model with dummy data.
    """
    print("Running Sanity Check for Question 1d: Encode")
    print ("-"*80)

    # Configure for Testing
    reinitialize_layers(model)
    source_lengths = [len(s) for s in src_sents]
    source_padded = model.vocab.src.to_input_tensor(
        src_sents, device=model.device)

    # Load Outputs
    enc_hiddens_target = torch.load(
        './sanity_check_en_es_data/enc_hiddens.pkl')
    dec_init_state_target = torch.load(
        './sanity_check_en_es_data/dec_init_state.pkl')

    # Test
    with torch.no_grad():
        enc_hiddens_pred, dec_init_state_pred = model.encode(source_padded, source_lengths)
    assert(np.allclose(enc_hiddens_target.numpy(), enc_hiddens_pred.numpy())), "enc_hiddens is incorrect: it should be:\n {} but is:\n{}".format(enc_hiddens_target, enc_hiddens_pred)
    print("enc_hiddens Sanity Checks Passed!")
    assert(np.allclose(dec_init_state_target[0].numpy(), dec_init_state_pred[0].numpy())), "dec_init_state[0] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[0], dec_init_state_pred[0])
    print("dec_init_state[0] Sanity Checks Passed!")
    assert(np.allclose(dec_init_state_target[1].numpy(), dec_init_state_pred[1].numpy())), "dec_init_state[1] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[1], dec_init_state_pred[1])
    print("dec_init_state[1] Sanity Checks Passed!")
    print ("-"*80)
    print("All Sanity Checks Passed for Question 1d: Encode!")
    print ("-"*80)


def question_1e_sanity_check(model, src_sents, tgt_sents, vocab):
    """ Sanity check for question 1e. 
        Compares student output to that of model with dummy data.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1e: Decode")
    print ("-"*80)

    # Load Inputs
    dec_init_state = torch.load('./sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load('./sanity_check_en_es_data/enc_masks.pkl')
    target_padded = torch.load('./sanity_check_en_es_data/target_padded.pkl')

    # Load Outputs
    combined_outputs_target = torch.load(
        './sanity_check_en_es_data/combined_outputs.pkl')

    # Configure for Testing
    reinitialize_layers(model)
    COUNTER = [0]
    def stepFunction(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
       dec_state = torch.load(
           './sanity_check_en_es_data/step_dec_state_{}.pkl'.format(COUNTER[0]))
       o_t = torch.load(
           './sanity_check_en_es_data/step_o_t_{}.pkl'.format(COUNTER[0]))
       COUNTER[0]+=1
       return dec_state, o_t, None
    model.step = stepFunction

    # Run Tests
    with torch.no_grad():
        combined_outputs_pred = model.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
    assert(np.allclose(combined_outputs_pred.numpy(), combined_outputs_target.numpy())), "combined_outputs is incorrect: it should be:\n {} but is:\n{}".format(combined_outputs_target, combined_outputs_pred)
    print("combined_outputs Sanity Checks Passed!")
    print ("-"*80)
    print("All Sanity Checks Passed for Question 1e: Decode!")
    print ("-"*80)

def question_1f_sanity_check(model, src_sents, tgt_sents, vocab):
    """ Sanity check for question 1f. 
        Compares student output to that of model with dummy data.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1f: Step")
    print ("-"*80)
    reinitialize_layers(model)

    # Inputs
    Ybar_t = torch.load('./sanity_check_en_es_data/Ybar_t.pkl')
    dec_init_state = torch.load(
        './sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load('./sanity_check_en_es_data/enc_masks.pkl')
    enc_hiddens_proj = torch.load('./sanity_check_en_es_data/enc_hiddens_proj.pkl')

    # Output
    dec_state_target = torch.load('./sanity_check_en_es_data/dec_state.pkl')
    o_t_target = torch.load('./sanity_check_en_es_data/o_t.pkl')
    e_t_target = torch.load('./sanity_check_en_es_data/e_t.pkl')

    # Run Tests
    with torch.no_grad():
        dec_state_pred, o_t_pred, e_t_pred= model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
    assert(np.allclose(dec_state_target[0].numpy(), dec_state_pred[0].numpy())), "decoder_state[0] is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[0], dec_state_pred[0])
    print("dec_state[0] Sanity Checks Passed!")
    assert(np.allclose(dec_state_target[1].numpy(), dec_state_pred[1].numpy())), "decoder_state[1] is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[1], dec_state_pred[1])
    print("dec_state[1] Sanity Checks Passed!")
    assert(np.allclose(o_t_target.numpy(), o_t_pred.numpy())), "combined_output is incorrect: it should be:\n {} but is:\n{}".format(o_t_target, o_t_pred)
    print("combined_output  Sanity Checks Passed!")
    assert(np.allclose(e_t_target.numpy(), e_t_pred.numpy())), "e_t is incorrect: it should be:\n {} but is:\n{}".format(e_t_target, e_t_pred)
    print("e_t Sanity Checks Passed!")
    print ("-"*80) 
    print("All Sanity Checks Passed for Question 1f: Step!")
    print ("-"*80)


def main():
    """ Main function. """
    args = docopt(__doc__)

    assert(torch.__version__ >= "1.10.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = read_corpus(
        './sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = read_corpus(
        './sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in batch_iter(
        train_data, batch_size=BATCH_SIZE, shuffle=True):
        src_sents = src_sents
        tgt_sents = tgt_sents
        break
    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    if args['1d']:
        question_1d_sanity_check(model, src_sents, tgt_sents, vocab)
    elif args['1e']:
        question_1e_sanity_check(model, src_sents, tgt_sents, vocab)
    elif args['1f']:
        question_1f_sanity_check(model, src_sents, tgt_sents, vocab)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
    
