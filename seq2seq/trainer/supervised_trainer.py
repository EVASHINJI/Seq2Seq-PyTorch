from __future__ import division
import os
import logging
import random
import time
import horovod.torch as hvd
import torch
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        model_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, 
                 model_dir='experiment',
                 best_model_dir='experiment/best',
                 loss=NLLLoss(), 
                 batch_size=64, 
                 random_seed=None,
                 checkpoint_every=100, 
                 print_every=100, 
                 max_epochs=5,
                 max_steps=10000, 
                 max_checkpoints_num=5, 
                 best_ppl=100000.0,
                 device=None):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.best_ppl = best_ppl
        self.max_checkpoints_num = max_checkpoints_num
        self.device = device
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, device=device)

        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.getcwd(), model_dir)
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.isabs(best_model_dir):
            best_model_dir = os.path.join(os.getcwd(), best_model_dir)
        self.best_model_dir = best_model_dir
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        self.model_checkpoints = []
        self.best_model_checkpoints = []

        self.logger = logging.getLogger(__name__)

    def save_model(self, model, steps, dev_ppl=None):
        model_fn = f"{steps}.pt"
        model_fp = os.path.join(self.model_dir, model_fn)

        # save model checkpoints
        while len(self.model_checkpoints) >= self.max_checkpoints_num:
            os.system(f"rm {self.model_checkpoints[0]}")
            self.model_checkpoints = self.model_checkpoints[1:]
        torch.save(model.state_dict(), model_fp)
        self.model_checkpoints.append(model_fp)

        # update checkpoints file
        with open(os.path.join(self.model_dir, "checkpoints"), 'w') as f:
            f.write('\n'.join(self.model_checkpoints[::-1]))

        # save best model checkpoints
        if dev_ppl and dev_ppl < self.best_ppl:
            self.logger.info(f"Best model dev ppl {dev_ppl}.")
            self.best_ppl = dev_ppl
            while len(self.best_model_checkpoints) >= self.max_checkpoints_num:
                os.system(f"rm {self.best_model_checkpoints[0]}")
                self.best_model_checkpoints = self.best_model_checkpoints[1:]
            
            best_model_fp = os.path.join(self.best_model_dir, model_fn)
            os.system(f"cp {model_fp} {best_model_fp}")
            self.best_model_checkpoints.append(best_model_fp)


    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        device = self.device
        log = self.logger
        max_epochs = self.max_epochs
        max_steps = self.max_steps

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        step = 0
        steps_per_epoch = len(data)
        start_epoch = (start_step - step) // steps_per_epoch
        step = start_epoch * steps_per_epoch
        for batch in data:
            if step >= start_step: break
            step += 1
        if start_epoch or start_step:
            logging.info(f"Resume from Epoch {start_epoch}, Step {start_step}")

        for epoch in range(start_epoch, max_epochs):
            model.train(True)
            for batch in data:
                step += 1
                src_variables = batch['src'].to(device)
                tgt_variables = batch['tgt'].to(device)
                src_lens = batch['src_len'].view(-1).to(device)
                tgt_lens = batch['tgt_len'].view(-1).to(device)

                # print(src_variables, src_lens, tgt_variables)
                # exit(0)

                loss = self._train_batch(src_variables, src_lens.tolist(), tgt_variables, model, teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = f"Process {100.0*(step%steps_per_epoch)/steps_per_epoch:.2f}% of Epoch {epoch}, Total step {step}, Train {self.loss.name} {print_loss_avg:.4f}" 
                    if hvd.rank() == 0:
                        log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0:
                    dev_loss = None
                    if dev_data is not None:
                        dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                        log_msg = f"Dev {self.loss.name}: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"
                        if hvd.rank() == 0:
                            log.info(log_msg)
                        model.train(mode=True)
                    if hvd.rank() == 0:
                        self.save_model(model, step, dev_ppl=dev_loss)
                
                if step >= max_steps:
                    break

            if step >= max_steps:
                if hvd.rank() == 0:
                    log.info(f"Finish max steps {max_steps} at Epoch {epoch}.")
                break

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = f"Finished Epoch {epoch}, Train {self.loss.name} {epoch_loss_avg:.4f}"
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += f", Dev {self.loss.name}: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)
            if hvd.rank() == 0:
                self.save_model(model, step, dev_ppl=dev_loss)
                log.info(log_msg)
                log.info(f"Finish Epoch {epoch}, Total steps {step}.")

    def train(self, model, data, start_step=0, dev_data=None, optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        
        if optimizer is None:
            optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
        self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, start_step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
