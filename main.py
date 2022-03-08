from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import time
import os
from transformers import get_scheduler
import random


class Args:
    def __init__(self, train_path, valid_path, batch_size, num_workers, epochs, lr, shuffle, gpu_num, vocab_rev):
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.vocab_rev = vocab_rev

        
class NewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, use_title=False):
        super().__init__()

        ### preprocessing
        data = pd.read_csv(data_path, index_col=0)[['content']]
        safe_len = data['content'].apply(lambda x: len(x.split(' ')) < 350)
        clean = []
        for line in data['content']:
            if len(line.split(' ')) > 350:
                truncated_line = ' '.join(line.split(' ')[:350])
                clean.append(truncated_line.lower())
        clean = pd.DataFrame(clean, columns=['content'])
        self.data = pd.concat([data[safe_len], clean], axis=0)        
        ### ### ### ### ### 
        self.tokenizer = tokenizer
        self.use_title = use_title
    def __getitem__(self, index):
        if self.use_title and random.random() < 0.01:
            text = self.data['title'].iloc[index]
        text = self.data['content'].iloc[index]
        # truncation 하면 전처리 대체 가능?
        inputs = self.tokenizer(f'<|endoftext|>{text}<|endoftext|>', return_tensors='pt')
        # inputs['labels'] = inputs['input_ids'][index]
        ids_len, mask_len = inputs['input_ids'].size(1), inputs['attention_mask'].size(1)
        return inputs, ids_len, mask_len
    def __len__(self):
        return len(self.data)

      
def NewsCollate(batch):
    batch, ids_len, mask_len = zip(*batch)
    max_ids, max_mask = max(ids_len), max(mask_len)
    ids_res, mask_res, label_res = [], [], []
    for i, sample in enumerate(batch):
        ids_pad = max_ids - ids_len[i]
        mask_pad = max_mask - mask_len[i]
        
        ids_tensor = torch.cat([sample['input_ids'], torch.LongTensor([[tokenizer.get_vocab()['<|endoftext|>']] * ids_pad])], dim=1)
        if want_to_change_vocab:
            ids_tensor = torch.cat([sample['input_ids'], torch.LongTensor([[tokenizer.get_vocab()['[PAD]']] * ids_pad])], dim=1)
        mask_tensor = torch.cat([sample['attention_mask'], torch.LongTensor([[0] * mask_pad])], dim=1)
        ids_res.append(ids_tensor)
        mask_res.append(mask_tensor)
        # label_res.append(sample['labels'].reshape(-1))
    ids_batch = torch.cat(ids_res, dim=0)
    mask_batch = torch.cat(mask_res, dim=0)
    return {'input_ids':ids_batch, 'attention_mask':mask_batch}

  
def get_logger(name: str, file_path: str, stream=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 시간, 로거 이름, 로깅 레벨, 메세지
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    # console에 출력하도록 설정
    stream_handler = logging.StreamHandler()
    # 현재 디렉토리에 파일로 로깅하도록 설정
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    # 현재 디렉토리에 로깅 저장
    logger.addHandler(file_handler)

    return logger

  
def save(filename, model, optimizer, logger):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    logger.info('Model saved')


def load(filename, model, optimizer, logger):
    # state = torch.load(filename)
    state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    logger.info('Model loaded : {}'.format(filename))
    
def train(model, tokenizer, dataloader, optimizer,
          lr_scheduler, epoch, train_begin, device):
    begin = epoch_begin = time.time()
    print_batch = 100

    total_num, total_batch_size = 0, len(dataloader)
    losses, batch_cnt = 0, 0
    print('train start...')
    for batch in dataloader:
        batch = {k:v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs[0]
        loss.backward()

        losses += loss.item()
        total_num += 1

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if batch_cnt % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0
            print('epoch: {:4d}, batch: {:5d}/{:5d}, lr: {:.16f},\nloss: {:.8f}, elapsed: {:6.2f}s {:6.2f}m {:6.2f}h'.format(
                epoch, batch_cnt, total_batch_size,
                optimizer.param_groups[0]['lr'],
                losses / total_num,
                elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        batch_cnt += 1
    print('train completed...')
    return losses / total_batch_size
  
def evaluate(model, tokenizer, dataloader, epoch, device):  
    model.eval()
    losses, batch_cnt = 0, 0
    print('evaluate start...')
    with torch.no_grad():
        for batch in dataloader:
            if batch_cnt % 500 == 0:
                batch = {k:v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs[0]
                print(f"loss {loss}")
                losses = loss.item()
            batch_cnt += 1
    print('train completed...')
    return losses / len(dataloader)
    
    
if __name__=='__main__':
  args = Args('/content/drive/MyDrive/Data/archive/articles1.csv',
            '/content/drive/MyDrive/Data/archive/articles2.csv',
            2,
            0,
            30,
            7e-4,
            True,
            0,
            False)
  
  ##### model load #####
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  # model = GPT2Model.from_pretrained('gpt2')
  want_to_change_vocab = args.vocab_rev
  pad_id = 50267 if want_to_change_vocab else tokenizer.eos_token_id
  # lm_head.weight 	 torch.Size([50257, 768]) 추가
  # loss func.이 내재되어 있음
  model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=pad_id) 
  
  # 애초에 attn mask가 있기 때문에 안해도 무방
  # 그러나 open-end generation을 피하는 한 가지 방법
  # '<|endoftext|>': 50256
  # 마구 추가하면 pretrain 방식과 다르기 때문에 성능을 저하시킬 수 있다
  if want_to_change_vocab:
      special_tokens =  {'pad_token': '[PAD]'}
                      # 'bos_token': '<|endoftext|>', 
                      # 'additional_special_tokens': ['[SP1]', '[SP2]']}
      tokenizer.add_special_tokens(special_tokens)
      vocab = tokenizer.get_vocab()
      # print(vocab)
      model.resize_token_embeddings(len(vocab))
  
  ##### dataset #####
  dataset = NewsDataset(args.train_path, tokenizer)
  dataloader = DataLoader(dataset,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          collate_fn=NewsCollate,
                          shuffle=args.shuffle)
  valid_dataset = NewsDataset(args.valid_path, tokenizer)
  valid_dataloader = DataLoader(dataset,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          collate_fn=NewsCollate)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

num_training_steps = args.epochs * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=8000,
    num_training_steps=num_training_steps
)

logger = get_logger(name='train',
                    file_path=os.path.join('.', 'train_log.log'),
                    stream=True)

#### START #####
n_epoch = 0
train_begin = time.time()
for epoch in range(args.epochs):
    train_loss = train(model, tokenizer, dataloader, optimizer, lr_scheduler, epoch, train_begin, args.device)
    logger.info('Epoch %d (Training) Loss %0.8f' % (epoch, train_loss))
    
    valid_loss = evaluate(model, tokenizer, valid_dataloader, epoch, args.device)
    
    model.eval()
    inputs = tokenizer.encode('''How serious is the presence of the Covid virus in deer for humans?"''', return_tensors='pt')
    sample_outputs = model.generate(
                            bos_token_id=inputs,
                            do_sample=True,   
                            top_k=50, 
                            max_length = 100,
                            top_p=0.90, 
                            num_return_sequences=3
                        )
    
    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    
    # make_directory('checkpoint')
    save(os.path.join('checkpoint', f"model_{epoch:03d}.pt"), model, optimizer, logger)

    epoch_end_time = time.time()
    n_epoch += 1
    print(f'For {(epoch_end_time - epoch_start_time)/60:6.2f}, {n_epoch} Epoch Finished')
  
