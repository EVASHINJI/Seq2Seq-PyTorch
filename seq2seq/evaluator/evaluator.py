from __future__ import print_function, division

import torch

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64, device=None):
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        device = self.device

        loss.reset()
        match = 0
        total = 0

        tgt_vocab = data.dataset.tgt_vocab
        pad = tgt_vocab.word2idx[tgt_vocab.pad_token]

        with torch.no_grad():
            for batch in data:
                src_variables = batch['src'].to(device)
                tgt_variables = batch['tgt'].to(device)
                src_lens = batch['src_len'].view(-1).to(device)
                tgt_lens = batch['tgt_len'].view(-1).to(device)

                decoder_outputs, decoder_hidden, other = model(src_variables, src_lens.tolist(), tgt_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = tgt_variables[:, step + 1]
                    loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
