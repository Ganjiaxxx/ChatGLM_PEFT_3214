import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm
from main.loader import AutoDataloader
from main.analysis import Analysis
from accelerate import Accelerator
accelerator = Accelerator()


class Trainer():

    def __init__(self, tokenizer, from_pretrained, loader_name, data_path, config=None, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.loader_name = loader_name
        self.data_path = data_path
        self.from_pretrained = from_pretrained
        self.data_path = data_path
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.eval_mode = eval_mode
        self.config_init()
        self.dataloader_init()
        self.model_init(resume_path=resume_path)
        self.analysis = Analysis()

    def config_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained, trust_remote_code=True) if self.config is None else self.config
        if self.config.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Llama3需要手动设置chat_template
            if self.tokenizer.chat_template is None:
                with open('chat_template/llama3_chat.jinja', 'r', encoding='utf-8') as f:
                    self.tokenizer.chat_template = f.read()


    def model_init(self, resume_path=None):
        if self.accelerate.is_local_main_process:
            print('AutoModel Choose Model: {}\n'.format(
                self.from_pretrained))
        if self.config.model_type == 'chatglm':
            target_modules=['query_key_value']
            self.model = AutoModel.from_pretrained(
                self.from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        elif self.config.model_type == 'llama':
            target_modules=["q_proj", "k_proj", "v_proj"]
            self.model = LlamaForCausalLM.from_pretrained(
                self.from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        elif self.config.model_type == 'qwen':
            target_modules=["c_attn", "c_proj", "w1", "w2"]
            self.model = AutoModelForCausalLM.from_pretrained(
                self.from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.config.model_type == 'qwen2':
            target_modules=["q_proj", "k_proj", "v_proj"]
            self.model = AutoModelForCausalLM.from_pretrained(
                self.from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            target_modules=target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(
                self.model, resume_path, config=peft_config)
        else:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def dataloader_init(self):
        d = AutoDataloader(self.tokenizer, self.config, loader_name=self.loader_name, data_path=self.data_path,
                           max_length=self.max_length)
        self.train_loader, self.eval_loader = d(
            self.batch_size, self.batch_size_eval, self.eval_mode, True)

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, eval_call_epoch=None, eval_call_step=None):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_epoch=eval_call_epoch, eval_call_step=eval_call_step,)

    def train(self, resume_step=None, num_epochs=30, lr=1e-4,  eval_call_epoch=None, eval_call_step=None):
        # 一些数据集会显存不足，才用以下的代码，例如weibo
        '''accumulation_steps = 2  # 每2小批次累积一次梯度
        num_update_steps_per_epoch = len(self.train_loader) // accumulation_steps
        total_train_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = int(0.03 * total_train_steps)  # 比如前 3% steps 用来 warmup'''

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        # 例如weibo
        #scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        self.model, optimizer, train_loader, scheduler = self.accelerate.prepare(
            self.model, optimizer, self.train_loader, scheduler)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0
            eval_scores = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }

            train_iter = tqdm(train_loader)
            self.model.train()

            for idx, it in enumerate(train_iter):

                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                # weibo数据集显存不够，所以这样做
                #loss = loss / accumulation_steps

                # loss.backward()
                self.accelerate.backward(loss)

                '''if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(self.train_loader):
                    optimizer.step()           # 更新权重 :contentReference[oaicite:6]{index=6}
                    scheduler.step()           # 学习率调度
                    #self.model.zero_grad()
                    optimizer.zero_grad()'''

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(logits, labels)
                for k, v in metrics.items():
                    eval_scores[k] += v

                train_loss += loss.data.item()
                #train_loss += loss.data.item() * accumulation_steps
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, **{k: v / train_count for k, v in eval_scores.items()})
                
                # 按照step保存模型
                if eval_call_step is not None and eval_call_step(train_step):
                    self.save_model(train_step)

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
                **{k: v / train_count for k, v in eval_scores.items()}
            })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    @accelerator.on_local_main_process
    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        save_path = f'./save_model/{dir}/ChatGLM_{current_step}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model = self.accelerate.unwrap_model(self.model)
        save_model.save_pretrained(
            save_path,
            is_main_process=self.accelerate.is_main_process,
            save_function=self.accelerate.save,
        )
        
        '''# unwrap 出的是 PeftModel（含 LoRA adapter）
        peft_model = self.accelerate.unwrap_model(self.model)
        # 只保存 adapter 权重和配置
        peft_model.save_pretrained(save_path, safe_serialization=True)
        # 如果需要也可以顺便保存 tokenizer
        if self.accelerate.is_main_process:
            self.tokenizer.save_pretrained(save_path)'''
        
        # 显式保存适配器配置
        '''self.model.save_pretrained(
            save_path,
            is_main_process=self.accelerate.is_main_process,
            safe_serialization=True,  # 强制使用safetensors格式
            state_dict=self.accelerate.get_state_dict(self.model),  # 关键参数
        )'''

        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, pure_eval=False):
        if pure_eval:
            self.model = self.accelerate.prepare_model(self.model)
        self.eval_loader = self.accelerate.prepare_data_loader(
            self.eval_loader)

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            eval_scores = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:

                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(logits, labels)
                for k, v in metrics.items():
                    eval_scores[k] += v

                eval_loss += loss.data.item()
                eval_count += 1

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, **{k: v / eval_count for k, v in eval_scores.items()})

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                **{k: v / eval_count for k, v in eval_scores.items()}
            })

    def compute_metrics(self, logits, labels):
        # logits是 bz x seq x vocab
        # 序列的每一个位置都会预测下一个token，每个都会预测vacab个token，值是对应的概率

        # 去掉了每个序列中最后的预测 ，为了跟labels对齐
        shift_logits = logits[..., :-1, :]
        # 得到最大概率的token
        pred_logits = shift_logits.argmax(-1)
        pred_logits = pred_logits.tolist()
        # labels去掉了第一个token，这样就可以和pred_logits对齐
        shift_labels = labels[..., 1:].tolist()

        metrics_dct = {'rouge-1': [], 'rouge-2': [],
                       'rouge-l': [], 'bleu-4': []}
        for pred_ids, label_ids in zip(pred_logits, shift_labels):
            try:
                answer_idx = 0
                # 寻找答案所在位置（也就是assitant的内容）
                for i in range(len(label_ids)):
                    if label_ids[i] != -100:
                        answer_idx = i
                        break
                # 只比对assitant内容，原本的logits是input_ids(包括user内容和assitant内容)然后去预测下一个token概率
                pred_ids = pred_ids[answer_idx:]
                label_ids = label_ids[answer_idx:]
                pred_txt = self.tokenizer.decode(pred_ids).strip()
                label_txt = self.tokenizer.decode(label_ids).strip()
                pred_tokens = list(jieba.cut(pred_txt))
                label_tokens = list(jieba.cut(label_txt))
                rouge = Rouge()
                scores = rouge.get_scores(
                    ' '.join(pred_tokens), ' '.join(label_tokens))
                for k, v in scores[0].items():
                    metrics_dct[k].append(round(v['f'] * 100, 4))
                metrics_dct['bleu-4'].append(
                    sentence_bleu(
                        [label_tokens],
                        pred_tokens,
                        smoothing_function=SmoothingFunction().method3,
                    )
                )
            except:
                continue
        return {k: np.mean(v) if len(v) > 0 else 0 for k, v in metrics_dct.items()}
