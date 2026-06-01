from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Tuple, List
import json
import torch
import os
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import json_repair
from time import perf_counter

class Predictor():
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 data_path="youku_1000_lora"
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
        self.data_present_path="/root/ChatGLM_PEFT_new/data/Tweebank_NER/present.json"
        self.data_path = data_path
        self.data_present = self.get_data_present(self.data_present_path)
        self.model = PeftModel.from_pretrained(
            self.model, resume_path, config=peft_config)
        self.model_to_device(gpu=num_gpus)

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
    
    
    def process_model_outputs(self, inputs, outputs):
        pred = []        # 一维字符串列表（一个batch中的）
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            response = self.tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
            # 修复json
            content = json_repair.loads(response)
            # 又转换为字符串
            content = json.dumps(content, ensure_ascii=False)
            pred.append(content)
        return pred
    
    def build_chat_input(self, query, history=None):
        if history is None:
            history = []
        history.append(query)
        max_input_tokens = 0
        new_batch_input = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        return new_batch_input, max_input_tokens
    
    def batchify(self, query, batch_size):
        return [query[i:i + batch_size] for i in range(0, len(query), batch_size)]
    

    def parse_entities(self, json_str: str) -> List[Tuple[str, str]]:
        """
        将 LLM 的输出或标注（JSON 字符串）解析为 (entity, type) 元组列表。
        """
        try:
            ents = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果非标准 JSON ，直接返回空
            return []
        result = []
        for ent in ents:
            if isinstance(ent, dict):
                text = ent.get("entity", "").strip()   # 如果是词性得改成pos
                typ  = ent.get("type", "").strip()
                if text and typ:
                    result.append((text, typ))
        return result

    def compute_ner_metrics(self,
        true_list: List[str],
        pred_list: List[str]
    ) -> dict[str, float]:
        """
        接收两列表：true_list 和 pred_list，每项都是 LLM 输出的 JSON 字符串；
        返回整体的 Precision, Recall, F1。
        """
        total_true = 0
        total_pred = 0
        total_correct = 0

        for true_json, pred_json in zip(true_list, pred_list):
            true_set = set(self.parse_entities(true_json))
            pred_set = set(self.parse_entities(pred_json))
            total_true += len(true_set)
            total_pred += len(pred_set)
            total_correct += len(true_set & pred_set)

        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall    = total_correct / total_true if total_true > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False, batch_size=2):
        test_path = self.data_present[self.data_path]['test']
        query = []      # 一维字符串列表（总的query)
        labels = []     # 一维字符串列表（总的labels）
        preds = []      # 一维字符串列表（总的预测）
        
        with open(test_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 解析每行JSON对象
                data = json.loads(line.strip())
                convs = data['conversations']
                for conv in convs:
                    if (conv["role"] == "user"):
                        query.append(conv["content"])
                    
                    elif (conv["role"] == "assistant"):
                        items = conv["content"]  # 字符串，例如"[{"entity":"...", "type": "..."}, {"entity":"...", "type": "..."}]"
                        labels.append(items)  # 不能使用extend，它会把字符串中的每一个字符逐个添加 
        
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        with torch.no_grad():
            # 对query进行分组，每组batch_size（最后一组大小可能小于batch_size)
            # 二维列表
            batches = self.batchify(query, batch_size)
            for batch in batches:
                if build_message:
                    inputs = []
                    batch_max_len = 0
                    for i, t in enumerate(batch):
                        if isinstance(t, str):
                            t = {'role': 'user', 'content': t}
                        if history is not None and len(history) > 0:
                            h_unit = history[i]
                        else:
                            h_unit = []
                        t, max_input_tokens = self.build_chat_input(t, h_unit)
                        if batch_max_len < max_input_tokens:
                            batch_max_len = max_input_tokens
                        inputs.append(t)
                else:
                    inputs = query
                    batch_max_len = 0
                    for i in range(len(query)):
                        if len(query[i]) > batch_max_len:
                            batch_max_len = len(query[i])

                start = perf_counter()
                # 一个batch中的inputs进行补齐得到batched_inputs
                batched_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",            # 补齐到左边
                    truncation=True).to(self.device)
                if self.config.model_type == 'llama':
                    batched_inputs = batched_inputs.data

                batched_outputs = self.true_model.generate(**batched_inputs, **{
                    'max_new_tokens': max_new_tokens,
                    'num_beams': num_beams,
                    'do_sample': do_sample,
                    'top_p': top_p,
                    "temperature": temperature,
                    "eos_token_id": self.eos_token_id
                })

                pred = self.process_model_outputs(batched_inputs, batched_outputs)  # pred是一个batch中的预测，是一维字符串列表
                preds.extend(pred)         # preds是一维字符串列表，而且是所有的预测输出
            
            metrics = self.compute_ner_metrics(labels, preds)
            print("=== NER Evaluation ===")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-score:  {metrics['f1']:.4f}")
            # 指定目标目录
            out_dir = "/root/ChatGLM_PEFT_new/data_record/predict_llama_TweebankNER_1000_lora1_50000"
            # 递归创建目录，若已存在则不报错
            os.makedirs(out_dir, exist_ok=True)

            end = perf_counter()
            with open(f"{out_dir}/time.txt", "w", encoding="utf-8") as f:
                f.write(f"Elapsed (perf_counter): {end - start:.6f} s")

            with open(f"{out_dir}/output_lora2.txt", "w", encoding="utf-8") as file:
                for s in preds:
                    file.write(s + "\n")  # 每个字符串末尾添加换行符
                    
            with open(f"{out_dir}/classification_report.txt", "w", encoding="utf-8") as f:
                f.write(f"Precision: {metrics['precision']:.4f}\n" + f"Recall:    {metrics['recall']:.4f}\n" + f"F1-score:  {metrics['f1']:.4f}\n")         
            

    def __call__(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True, batch_size = 2):
        self.predict(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message, batch_size)
