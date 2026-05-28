import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer
from typing import List
import copy

class LLMDPODataset(Dataset):

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer, config, file_name, max_length=512, do_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.data = self.load_jsonl(file_name)
        self.random_list = [idx for idx in range(len(self.data))]
        if self.do_shuffle:
            random.shuffle(self.random_list)

    def load_jsonl(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return data

    def init_ids_and_masks(self):
        if self.config.model_type == 'llama':
            # Llama 2
            if "<|begin_of_text|>" not in self.tokenizer.special_tokens_map.values():
                return [], []
            # Llama 3
            else:
                return [self.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')], [-100]
        else:
            return [], []

    def apply_template_with_char_tokenize(self, messages: List[dict], content_role="assistant", add_generation_prompt=False):
        """
        使用 tokenizer.apply_chat_template 生成模板字符串，但把 messages 中
        指定 role 的 content 按“字符粒度”单独编码，再把编码结果插回模板中。

        返回：token id 列表（list[int]）
        """
        # 找出要按字符编码的那条消息（这里假设只有一条 content 需要按字符编码）
        # 如果有多条需要逐条处理，可把逻辑扩展为遍历 messages
        # 取原始 content
        target_msg = None
        for m in messages:
            if m.get("role") == content_role:
                target_msg = m
                break

        if target_msg is None:
            raise ValueError(
                f"No message with role={content_role} found in messages")

        orig_content = target_msg["content"]

        # 用一个独一无二的占位符替换 content（确保占位符在模板里不会被其他文本干扰）
        placeholder = "<<<__CONTENT_PLACEHOLDER_7f3b__>>>"

        # 构造临时消息列表：把目标消息的 content 用占位符替代
        tmp_msgs = []
        for m in messages:
            if m is target_msg:
                tmp_msgs.append({"role": m["role"], "content": placeholder})
            else:
                tmp_msgs.append(m)

        # 获取带模板的字符串（不 token 化）
        # 有些 tokenizer 实现可能接受不同参数名，这里使用常见参数 add_generation_prompt 控制是否添加 generation hint
        template_text = self.tokenizer.apply_chat_template(
            tmp_msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

        # 把模板字符串按占位符拆成前缀和后缀
        if placeholder not in template_text:
            # 兜底：如果没有按预期出现占位符，直接 raise，便于调试
            raise RuntimeError(
                "Placeholder not found in template_text. Template output:\n" + template_text)

        prefix_text, suffix_text = template_text.split(placeholder, 1)

        # 对 prefix 和 suffix 各自编码（不添加 special tokens）
        # 使用 tokenizer.encode 或 tokenizer.__call__ 都可，确保不额外加 BOS/EOS
        prefix_ids = self.tokenizer.encode(
            prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(
            suffix_text, add_special_tokens=False)

        # 对内容按字符逐个编码，注意每个字符的编码可能是多个 token id（取决于分词器）
        char_ids = []
        # llama一个汉字可能对于多个token 所以不能用[0]，且不用列表推导式，保持tokens_to_train不嵌套
        for ch in orig_content:
            # 避免对空字符进行编码（虽然通常不会有）
            if ch == "":
                continue
            # 不添加 special tokens，确保只得到字符本身的 token ids
            ch_tok_ids = self.tokenizer.encode(ch, add_special_tokens=False)
            char_ids.extend(ch_tok_ids)

        # 最终拼接
        gjx_ids = prefix_ids + char_ids + suffix_ids
        gjx = self.tokenizer.decode(gjx_ids)
        return gjx_ids

    def build_single_message_dpo(self, t):
        role = t['role']
        # user的话 直接编码
        if (role == "user"):
            ids = self.tokenizer.apply_chat_template([t])
        # assistant的话 会对其content部分一个一个汉字进行编码
        elif (role == "assistant"):
            ids = self.apply_template_with_char_tokenize(
                [t], content_role=role, add_generation_prompt=False)

        # 不管什么模型 只要不是assistant就直接-100返回
        if t['role'] in ['user', 'system']:
            ls = [-100 for _ in ids]
            return ids, ls

        # assistant的内容
        # 如果是llama需要mask掉除content以外的内容
        if self.config.model_type == 'llama':
            # 由于只需要训练生成的回答，因此要mask掉最后一组对话的身份信息以及无关的符号
            # Llama 2
            if "<|begin_of_text|>" not in self.tokenizer.special_tokens_map.values():
                tokens_to_train = []
                # llama一个汉字可能对于多个token 所以不能用[0]，且不用列表推导式，保持tokens_to_train不嵌套
                for char in t['content']:
                    token_ids = self.tokenizer.encode(
                        char, add_special_tokens=False)
                    tokens_to_train.extend(token_ids)
                tokens_to_train.extend(
                    [self.tokenizer.convert_tokens_to_ids('</s>')])
            # Llama 3
            else:
                tokens_to_train = []
                # llama一个汉字可能对于多个token 所以不能用[0]，且不用列表推导式，保持tokens_to_train不嵌套
                for char in t['content']:
                    token_ids = self.tokenizer.encode(
                        char, add_special_tokens=False)
                    tokens_to_train.extend(token_ids)
                tokens_to_train.extend(
                    [self.tokenizer.convert_tokens_to_ids('<|eot_id|>')])

            ls = [-100] * (len(ids) - len(tokens_to_train)) + tokens_to_train

        # 对于其他模型，不需要对最后一组信息进行局部mask，将最后一组信息全部作为训练对象
        else:
            ls = ids

        return ids, ls

    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item

        # input_ids, labels = [], []
        input_ids, labels = self.init_ids_and_masks()
        chosen_full_tokens, chosen_mask = self.init_ids_and_masks()
        rejected_full_tokens, rejected_mask = self.init_ids_and_masks()

        for t in conv:
            ids, ls = self.build_single_message_dpo(t)
            input_ids.extend(ids)
            labels.extend(ls)

            role = t['role']
            if (role == "user"):
                prompt = ids
                prompt_ls = ls
            elif (role == "assistant"):
                # 为了获取{"role": "assistant", "content":...}模板
                template_chosen = copy.deepcopy(t)
                template_rejected = copy.deepcopy(t)

        chosen = item['gold_answers']
        rejected = item['bad_answers']

        # 修改模板的content部分
        template_chosen['content'] = chosen
        template_rejected['content'] = rejected

        # 构造成模板，送进去函数
        ids1, ls1 = self.build_single_message_dpo(template_chosen)
        '''chosen_full_tokens = [
            self.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')]'''
        chosen_full_tokens.extend(prompt)
        chosen_full_tokens.extend(ids1)
        #chosen_mask = [False]
        chosen_mask.extend(prompt_ls)  # prompt_ls全是-100
        chosen_mask.extend(ls1)        # ls1除了content部分也全是-100

        # 构造成模板，送进去函数
        ids2, ls2 = self.build_single_message_dpo(template_rejected)
        '''rejected_full_tokens = [
            self.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')]'''
        rejected_full_tokens.extend(prompt)
        rejected_full_tokens.extend(ids2)
        #rejected_mask = [False]
        rejected_mask.extend(prompt_ls)
        rejected_mask.extend(ls2)

        '''role = t['role']
            # user部分整体进行编码            
            if (role == "user"):
                ids = self.tokenizer.apply_chat_template([t])
                prompt = ids
            # assistant部分每个汉字单独进行编码
            elif (role == "assistant"):
                # 对应[gMASK] <sop> <|assistant|> \n
                ids = [151331, 151333, 151337, 198]
                ids.extend([
                    self.tokenizer.encode(char, add_special_tokens=False)[0]
                    for char in t['content']
                ])
            ls = ids if role not in ['user', 'system'] else [-100 for _ in ids]
            input_ids.extend(ids)
            labels.extend(ls)'''

        '''chosen = item['gold_answers']
        rejected = item['bad_answers']
        # 对应[gMASK] <sop> <|assistant|> \n
        ids1 = [151331, 151333, 151337, 198]
        ids1.extend([
            self.tokenizer.encode(char, add_special_tokens=False)[0]
            for char in chosen
        ])
        chosen_full_tokens = []
        chosen_full_tokens.extend(prompt)
        chosen_full_tokens.extend(ids1)
        # 对应[gMASK] <sop> <|assistant|> \n
        ids2 = [151331, 151333, 151337, 198]
        ids2.extend([
            self.tokenizer.encode(char, add_special_tokens=False)[0]
            for char in rejected
        ])
        rejected_full_tokens = []
        rejected_full_tokens.extend(prompt)
        rejected_full_tokens.extend(ids2)'''

        max_length = self.max_length
        prompt = prompt[:max_length]
        chosen_full_tokens = chosen_full_tokens[:max_length]
        rejected_full_tokens = rejected_full_tokens[:max_length]
        chosen_mask = chosen_mask[:max_length]
        rejected_mask = rejected_mask[:max_length]
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        return {'prompt': prompt, 'chosen': chosen_full_tokens, 'rejected': rejected_full_tokens, 'chosen_mask': chosen_mask, 'rejected_mask': rejected_mask, 'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, index):
        index = self.random_list[index]
        data = self.data[index]
        prompt, chosen, rejected, chosen_mask, rejected_mask, input_ids, labels = self.process_item(
            data).values()

        prompt = torch.tensor(prompt)
        chosen = torch.tensor(chosen)
        rejected = torch.tensor(rejected)
        chosen_mask = torch.tensor(chosen_mask)
        rejected_mask = torch.tensor(rejected_mask)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        # 使得-100的地方变成0，其它有效部分变成1
        chosen_mask = chosen_mask != -100
        rejected_mask = rejected_mask != -100

        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_mask': chosen_mask,
            'rejected_mask': rejected_mask,
            'input_ids': input_ids,
            'labels': labels
        }

    def __len__(self):
        return len(self.data)
