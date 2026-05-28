from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Tuple, List
import json
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import re


class Predictor():
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 label_token_len=2, data_path = "youku_1000_nt2"
                 ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model_from_pretrained = model_from_pretrained
        self.config = AutoConfig.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        
        if self.config.model_type == 'chatglm':
            self.model = AutoModel.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            self.eos_token_id = self.config.eos_token_id
        elif self.config.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            #self.eos_token_id = terminators
            if not hasattr(self, 'eos_token_id'):
                self.eos_token_id = []
            for t in terminators:
                if t is not None:
                    self.eos_token_id.append(t)
        elif self.config.model_type == 'qwen':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.config.model_type == 'qwen2':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        
        self.model = PeftModel.from_pretrained(
            self.model, resume_path, config=peft_config)
        self.candidates = {}                      # 键标签和值token 
        self.label_token_len = label_token_len    # 标签对应的Token长度（每个标签都一致）
        self.data_present_path="./data/youku/present.json"
        self.data_path = data_path
        self.data_present = self.get_data_present(self.data_present_path)
        self.model_to_device(gpu=num_gpus)
        self.labels_init()

    def model_to_device(self, gpu=[0]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(self.device)
        self.true_model = self.model.module if hasattr(
            self.model, 'module') else self.model
        
    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present
    
    def labels_init(self):
        lables_path = self.data_present[self.data_path]['labels']
        with open(lables_path, 'r', encoding='utf-8') as f:
            self.candidates = json.load(f)

        sample_list = next(iter(self.candidates.values()))
        self.label_token_len = len(sample_list)
    
    def process_preds(self, input_idx):
        pred = []   # 一维字符串列表（一个batch中的）
        for input_id in input_idx:
            # skip_special_tokens如果为True会跳过开头的'[gMASK] <sop> [gMASK] <sop> <|user|> \n'
            # 以及跳过输出的开头' <|assistant|> '
            response = self.tokenizer.decode(input_id, skip_special_tokens=False).strip()
            # 只获取' <|assistant|> '后面的内容，即为模型输出部分
            if (self.config.model_type == 'chatglm'):
                parts = response.split(' <|assistant|> ')
            elif (self.config.model_type == 'llama'):
                parts = response.split('assistant<|end_header_id|>\n\n')
            content = parts[-1].strip() if len(parts) > 1 else ''
            pattern = re.compile(r'(.+?)\(([^()]+)\)')
            items = pattern.findall(content)
            for _, item in items:
                if (item == "不是实体"):
                    pred.append("O")
                else:
                    pred.append(item)
    
        return pred
    
    def process_model_outputs(self, input_idx):
        pred = []   # 一维字符串列表（一个batch中的）

        for input_id in input_idx:
            # skip_special_tokens如果为True会跳过开头的'[gMASK] <sop> [gMASK] <sop> <|user|> \n'
            # 以及跳过输出的开头' <|assistant|> '
            response = self.tokenizer.decode(input_id, skip_special_tokens=False).strip()
            # 只获取' <|assistant|> '后面的内容，即为模型输出部分
            if (self.config.model_type == 'chatglm'):
                parts = response.split(' <|assistant|> ')
            elif (self.config.model_type == 'llama'):
                parts = response.split('assistant<|end_header_id|>\n\n')
            content = parts[-1].strip() if len(parts) > 1 else ''
            pred.append(content)
        return pred
    
    def build_chat_input(self, query, history=None):
        if history is None:
            history = []
        #[{'role': 'user', 'content': t}]
        history.append(query)
        max_input_tokens = 0

        # add_generation_prompt=True 会在末尾添加 <|assistant|> 内容，引导模型生成后续内容
        new_batch_input = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
         
        # 对"输入: "后面的序列一个一个字地编码
        content = history[-1]['content']
        input_sequence = content.split(" 输入: ")[1].strip()
        # 对于中文：chatglm4不会分词，所以之前用[0]，但llama会分词，所以也得像英文这样设置了
        # 这里对于单条句子会得到二维的列表，例如[[20565], [101961], [21], [103319], [13, 99698, 24], [98360], [16, 14, 17]]
        input_sequence = [
            self.tokenizer.encode(char, add_special_tokens=False)  
            for char in input_sequence
        ]
        max_char_num = 0
        max_char_num = max(max_char_num, len(input_sequence))
        return new_batch_input, max_input_tokens, input_sequence, max_char_num
    
    def batchify(self, query, batch_size):
        return [query[i:i + batch_size] for i in range(0, len(query), batch_size)]


    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False, batch_size=2):
        test_path = self.data_present[self.data_path]['test']
        preds2 = []     # 一维字符串列表（总的输出预测）（不仅有标签还有前面的汉字字符） 
        
        query = []      # 一维字符串列表（总的query)
        labels = []     # 一维字符串列表（总的labels）
        preds = []      # 一维字符串列表（总的预测）(只有标签)
        
        with open(test_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 解析每行JSON对象
                data = json.loads(line.strip())
                convs = data['conversations']
                for conv in convs:
                    if (conv["role"] == "user"):
                        query.append(conv["content"])
                    
                    elif (conv["role"] == "assistant"):
                        pattern = re.compile(r'(.+?)\(([^()]+)\)')
                        items = pattern.findall(conv["content"])
                        label = [] 
                        for _, item in items:
                            if (item == "不是实体"):
                                label.append("O")
                            else:
                                label.append(item)
                        labels.extend(label)    # 用的是extend，所以labels还是一维列表
                    
        
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        with torch.no_grad():
            colon_ids = self.tokenizer.encode("(", add_special_tokens=False)  
            comma_ids = self.tokenizer.encode(")", add_special_tokens=False)  
            # 对query进行分组，每组batch_size（最后一组大小可能小于batch_size)
            # 二维列表
            batches = self.batchify(query, batch_size)
            for batch in batches:
                if build_message:
                    inputs = []
                    input_sequence = []
                    batch_max_len = 0
                    max_char_num = 0
                    char_num_list = []
                    for i, t in enumerate(batch):
                        if isinstance(t, str):
                            t = {'role': 'user', 'content': t}
                        if history is not None and len(history) > 0:
                            h_unit = history[i]
                        else:
                            h_unit = []
                        t, max_input_tokens, sequence, char_num = self.build_chat_input(t, h_unit)
                        if batch_max_len < max_input_tokens:
                            batch_max_len = max_input_tokens
                        if max_char_num < char_num:
                            max_char_num = char_num
                        inputs.append(t)
                        # 三维列表，例如[[[20565],[13, 99698, 24], [98360], [16, 14, 17]], [[2685, 819], [10689],[37592], [7], [6192, 6334]]]
                        input_sequence.append(sequence)
                        char_num_list.append(char_num)
                else:
                    inputs = query
                    batch_max_len = 0
                    for i in range(len(query)):
                        if len(query[i]) > batch_max_len:
                            batch_max_len = len(query[i])
                # 一个batch中的inputs进行补齐得到batched_inputs
                batched_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",            # 补齐到左边
                    truncation=True).to(self.device)
                if self.config.model_type == 'llama':
                    batched_inputs = batched_inputs.data

                # 二维tensor==>二维list
                input_idx = batched_inputs["input_ids"].tolist()
                # batch_size原始的值是指定的，但是可能一个batch中的数量不足batch_size，所以这里得进行修改，按实际情况而定
                batch_size = len(input_idx)
                # 初始化完成状态跟踪器 (batch_size,)
                is_completed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                # 每个样本一个 logits 列表
                #logits_collection = [[] for _ in range(batch_size)]  
                for seq_idx in range(max_char_num):
                    #max_input_len = 0
                    # 拼接汉字以及固定的(
                    for b in range(batch_size):
                        if seq_idx < char_num_list[b]:
                            # 当前汉字对应的 token id（可能会有多个token）
                            char_token = input_sequence[b][seq_idx]   # 不要用切片，直接用索引，这样才能得到一维list，例如[13, 99698, 24]
                            # 更新 input_idx[b]（全部在 CPU 上，是一个 list），直接 extend 两个 list
                            input_idx[b].extend(char_token)    # 这样即使一个单词对应多个Token，也能一次性加入
                            input_idx[b].extend(colon_ids)
                            #if len(input_idx[b]) > max_input_len :
                            #    max_input_len = len(input_idx[b])
                        # 检查终止条件
                        else:
                            is_completed[b] = True

                    # 获取未完成样本的索引
                    # 活跃的样本在原本的batch中的索引
                    active_indices = torch.nonzero(~is_completed, as_tuple=False).squeeze(-1).tolist()
                    if len(active_indices) == 0:
                        # 如果全部完成则退出当前序列循环
                        break

                    # 每一个输入的汉字对应输出self.label_token_len个标签token
                    for step in range(self.label_token_len):
                        # 仅处理活跃样本
                        # 得再次对活跃样本进行对齐（因为每个样本添加的token个数会不一样）
                        pad_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
                        max_len = max(len(input_idx[i]) for i in active_indices)

                        padded_tensors = []
                        for orig_idx in active_indices:
                            seq = input_idx[orig_idx]
                            t = torch.tensor(seq, dtype=torch.long, device=self.device)
                            pad_len = max_len - t.size(0)
                            if pad_len > 0:
                                pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long, device=self.device)
                                t = torch.cat([pad_tensor, t], dim=0)
                            padded_tensors.append(t)
                        
                        active_inputs = torch.stack(padded_tensors, dim=0)
                        outputs = self.model(input_ids=active_inputs)   # 得传递二维张量
                        logits = outputs.logits  # [bz, seq_len, vocab_size]
                        # 取最后一个 token 的 logits作为当前预测
                        # 得到的logits的bz大小是活跃样本的数量，不是原本整个batch的大小
                        last_logits = logits[:, -1, :]  # [bz, vocab_size]
                        probs = F.softmax(last_logits, dim=-1)

                        # 对 活跃样本 中每个样本独立选取候选标签该 step 的 token
                        for idx, orig_batch_idx in enumerate(active_indices):
                            candidate_probs = {}
                            # 遍历每个候选标签
                            for label, token_seq in self.candidates.items():
                                candidate_token = token_seq[step]
                                token_prob = probs[idx, candidate_token].item()
                                candidate_probs[label] = token_prob

                            # 从候选中选择概率最大的token
                            selected_label = max(candidate_probs, key=candidate_probs.get)
                            selected_token = self.candidates[selected_label][step]
                            #selected_prob = candidate_probs[selected_label]

                            #print(f"Step {step}: orig_batch_idx {orig_batch_idx} 选择候选标签 '{selected_label}' 的 token {selected_token} (概率 {selected_prob:.5f})")

                            # 将选择的 token 追加到 input_idx 上（在 CPU 的 list 上直接 append 整数）
                            input_idx[orig_batch_idx].append(selected_token)
                        
                    # 例如'黄(人物名称)晓(人物名称)'   
                    # 这种情况，batch中每一个样本的input_idx最后''不会''多添加两个token
                    for orig_batch_idx in active_indices:
                        input_idx[orig_batch_idx].extend(comma_ids)
                
                preds2.extend(self.process_model_outputs(input_idx))
                #print(preds2)
                
                pred = self.process_preds(input_idx)    # pred是一个batch中的预测，是一维字符串列表
                preds.extend(pred)         # 使用extend，所以preds还是一维字符串列表，而且是所有的预测输出
            
            # 只看非 “O” 的实体类别
            entity_labels = sorted(set(labels) - {"O"})
            report = classification_report(
                y_true=labels,
                y_pred=preds,
                labels=entity_labels,        # 明确告诉 sklearn 只关心哪几个类别
                target_names=entity_labels,  # 与 labels 顺序、长度严格对应
                digits=5
            )
            print(report)

            # 指定目标目录
            out_dir = "./data_record/predict_youku_1000_dpo_beta0.2"
            # 递归创建目录，若已存在则不报错
            os.makedirs(out_dir, exist_ok=True)

            with open(f"{out_dir}/output_nt2.txt", "w", encoding="utf-8") as file:
                for s in preds2:
                    file.write(s + "\n")  # 每个字符串末尾添加换行符
            
            with open(f"{out_dir}/classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            


    def __call__(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True, batch_size = 2):
        self.predict(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message, batch_size)
