from metrics.evaluate import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BiaffineNER
from dataloader import get_useful_ones, get_mask
import os

def get_pred_entity(cate_pred, span_scores,label_set, is_flat_ner= True):
    top_span = []
    cate_pred_upper = torch.triu(cate_pred, diagonal=0)
    index_item = torch.where(cate_pred_upper == 1)
    row = index_item[0]
    col = index_item[1]
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        tmp = (label_set[cate_pred[i][j].item()], i, j,span_scores[i][j].item())
        top_span.append(tmp)

    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    
    if not top_span:
        top_span = [('ANSWER', 0, 0)]

    return top_span[0]

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.save_folder = args.save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.model = BiaffineNER(args=args)
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.best_score = 0
        self.label_set = train_dataset.label_set

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=train_sampler,
                                      batch_size=self.args.batch_size, num_workers=16)

        total_steps = len(train_dataloader) * self.args.num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        for epoch in range(self.args.num_epochs):
            train_loss = 0
            print('EPOCH:', epoch)
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'first_subword': batch[2],
                          'char_ids': batch[4],
                          }
                label = batch[-1]
                seq_length = batch[3]
                self.model.zero_grad()

                output = self.model(**inputs)
                optimizer.zero_grad()
                mask = get_mask(max_length=self.args.max_seq_length, seq_length=seq_length)
                mask = mask.to(self.device)
                tmp_out, tmp_label = get_useful_ones(output, label, mask)

                loss = loss_func(tmp_out, tmp_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # norm gradient
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=self.args.max_grad_norm)

                # update learning rate
                scheduler.step()
            print('train loss:', train_loss / len(train_dataloader))
            self.eval('dev')

    def eval(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset=dataset, sampler=eval_sampler, batch_size=self.args.batch_size,
                                     num_workers=16)

        self.model.eval()
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        eval_loss = 0
        outputs = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'first_subword': batch[2],
                      'char_ids': batch[4],
                      }
            label = batch[-1]
            seq_length = batch[3]
            with torch.no_grad():
                output = self.model(**inputs)
                for i in range(len(output)):
                    true_len = seq_length[i]
                    out = output[i][:true_len, :true_len]
                    
                    input_tensor, cate_pred = out.max(dim=-1)
                    label_pre = get_pred_entity(cate_pred, input_tensor, self.label_set, True)
                    # print(label_pre)
                    outputs.append(label_pre)

            mask = get_mask(max_length=self.args.max_seq_length, seq_length=seq_length)
            mask = mask.to(self.device)

            tmp_out, tmp_label = get_useful_ones(output, label, mask)
            loss = loss_func(tmp_out, tmp_label)
            eval_loss += loss.item()

        exact_match, f1 = evaluate(outputs, mode)

        print()
        print(exact_match)
        print(f1)

        if f1 > self.best_score:
            self.save_model()
            self.best_score = f1

    def save_model(self):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      }
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        torch.save(checkpoint, path)
        torch.save(self.args, os.path.join(self.args.save_folder, 'training_args.bin'))
        
    def load_model_train(self):
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def load_model_dev(self):
        path = os.path.join(self.save_folder, 'checkpoint.pth')
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
