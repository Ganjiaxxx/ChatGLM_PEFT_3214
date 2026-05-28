from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics import classification_report
import debugpy
import pdb
from typing import List, Tuple

# 不带思考过程 对应的奖励函数如下：

def content_reward(completions, answer, **kwargs):
    model = SentenceTransformer('/home/paraphrase-MiniLM-L6-v2')  
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    for text, ans in zip(responses, answer):
        emb1 = model.encode(text, convert_to_tensor=True)
        emb2 = model.encode(ans, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        reward = (sim+1.0)/2     # 余弦相似度范围-1~1 ==> 0~2 ==> 0~1
        rewards.append(reward)
    return rewards

def format_reward(completions, answer, **kwargs):
    # 段落模式：一个汉字 + 括号包裹至少一个汉字
    segment_pattern = re.compile(r'(.)\(([^()]+)\)')
    # 完全匹配模式：从头到尾都由若干这样的“段”拼接而成
    full_pattern = re.compile(r'^(?:.\([^()]+\))+$')
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    for text, ans in zip(responses, answer):
        # 1) 标准段数：从答案中找出所有合法段
        standard_segments = len(segment_pattern.findall(ans))
        # 2) 在生成文本中检查两个条件
        # 条件1：完全匹配格式
        cond1 = bool(full_pattern.match(text))
        # 条件2：段数一致
        generated_segments = len(segment_pattern.findall(text))
        cond2 = (generated_segments == standard_segments)
        
        # 3) 计算 reward
        if cond1 and cond2:
            reward = 1.0
        elif cond1 or cond2:
            reward = 0.5
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

# 带思考过程 对应的奖励函数如下：

# 严格格式要求：^$要求整行匹配 
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 稍微宽松的格式要求：1.不需要整行匹配 2.换行符也不需要完全匹配 3.</reasoning>和<answer>之间可以是任意的空白符（换行，tab，空格）
def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 计算文本中<reasoning></reasoning>和<answer></answer>标签的出现次数，并根据它们的位置和频率分配奖励。
def count_xml(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# 该函数用于计算每个模型输出的XML结构符合度，并返回奖励分数
def xmlcount_reward_func(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def extract_xml_answer(text: str) -> str:
    """
    从文本中提取第一对 <answer>...</answer> 之间的内容。
    如果找不到，返回空字符串。
    """
    start_tag = "<answer>"
    end_tag = "</answer>"

    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag))

    if start == -1 or end == -1:
        return ""

    return text[start + len(start_tag):end].strip()



# answer内部的格式要求：汉字(标签)
def think_format_reward(completions, answer, **kwargs):
    '''print("completions:")
    print(completions)'''
    responses = [completion[0]['content'] for completion in completions]
    '''print("responses:")
    print(responses)'''
    # 提取answer标签之间的内容
    extracted_responses = [extract_xml_answer(r) for r in responses]
    '''print("extracted_responses:")
    print(extracted_responses)
    print("answer:")
    print(answer)'''
    # 段落模式：一个字符 + 括号包裹至少一个字符
    segment_pattern = re.compile(r'(.)\(([^()]+)\)')
    # 完全匹配模式：从头到尾都由若干这样的“段”拼接而成
    full_pattern = re.compile(r'^(?:.\([^()]+\))+$')

    rewards = []
    # 有多少个回答，就对应有多少个重复的正确答案
    for text, ans in zip(extracted_responses, answer):
        # 1) 标准段数：从答案中找出所有合法段
        standard_segments = len(segment_pattern.findall(ans))
        # 2) 在生成文本中检查两个条件
        # 条件1：完全匹配格式
        cond1 = bool(full_pattern.match(text))
        # 条件2：段数一致
        generated_segments = len(segment_pattern.findall(text))
        cond2 = (generated_segments == standard_segments)
        
        # 3) 计算 reward
        if cond1 and cond2:
            reward = 0.5
        elif cond1 or cond2:
            reward = 0.25
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def think_content_reward(completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    # 提取answer标签之间的内容
    extracted_responses = [extract_xml_answer(r) for r in responses]

    model = SentenceTransformer('/home/paraphrase-MiniLM-L6-v2')  
    rewards = []
    for text, ans in zip(extracted_responses, answer):
        emb1 = model.encode(text, convert_to_tensor=True)
        emb2 = model.encode(ans, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        reward = sim+1.0     # 余弦相似度范围-1~1 ==> 0~2 
        rewards.append(reward)
    return rewards


def map_by_char_sequence(gen_text: str, ref_chars: list[str]) -> list[tuple[str, str]]:
    """
    从 gen_text 中按顺序提取标签，映射到 ref_chars 对应位置。
    
    Args:
      gen_text: 模型生成的字符串，例如 "张(人物名称)三(人物名称)李"
      ref_chars: 参考答案的字符序列，例如 ["张", "三", "李"]
    
    Returns:
      一个 list，形式为 [(char1, tag1), (char2, tag2), ...]
      如果某个字符未能匹配，则对应的 tag 为空字符串。
    """
    positions = []
    search_start = 0
    for ch in ref_chars:
        pos = gen_text.find(ch, search_start)
        if pos == -1:
            positions.append(None)
        else:
            positions.append(pos)
            search_start = pos + 1

    # 构造映射（使用列表而非 dict）
    result = []
    n = len(ref_chars)
    text_len = len(gen_text)
    for i, ch in enumerate(ref_chars):
        pos = positions[i]
        if pos is None:
            result.append((ch, "O"))
            continue
        # 找下一个字符的位置
        next_pos = text_len
        for j in range(i + 1, n):
            if positions[j] is not None:
                next_pos = positions[j]
                break
        # 切片提取标签
        tag_str = gen_text[pos + 1: next_pos].strip()
        tag_str = tag_str.replace("(", "").replace(")", "")
        result.append((ch, tag_str))

    return result


def extract_ref_chars_and_tags(answer_str: str):
    """
    从标准答案字符串中提取序列和对应标签列表。
    例如 "张(人物名称)三(人物名称)李(人物名称)四(人物名称)" 
    返回 (["张","三","李","四"], ["人物名称","人物名称","人物名称","人物名称"])
    """
    pairs = re.findall(r'(.)\(([^()]+?)\)', answer_str)
    ref_chars, ref_tags = zip(*pairs) if pairs else ([], [])
    return list(ref_chars), list(ref_tags)


def think_F1_content_reward(completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    # 提取answer标签之间的内容
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    print("extracted_responses")
    print(extracted_responses)
    print("answer")
    print(answer)

    for text, ans in zip(extracted_responses, answer):
        # 从标准答案字符串中提取汉字序列和对应标签列表
        ref_chars, ref_tags = extract_ref_chars_and_tags(ans)
        # 将正确答案的标签列表中“不是实体”全改成O
        labels = [ "O" if x == "不是实体" else x for x in ref_tags]

        mapping = map_by_char_sequence(text, ref_chars)
        preds = [tag for _, tag in mapping]

        # 只看非 “O” 的实体类别
        entity_labels = sorted(set(labels) - {"O"})
        report = classification_report(
            y_true=labels,
            y_pred=preds,
            labels=entity_labels,        # 明确告诉 sklearn 只关心哪几个类别
            target_names=entity_labels,  # 与 labels 顺序、长度严格对应
            digits=5,
            output_dict=True
        )
        micro_f1 = report['micro avg']['f1-score']
        rewards.append(micro_f1)
    return rewards


def token_level_reward(preds: list[str], labels: list[str]) -> float:
    assert len(preds) == len(labels)
    rewards = [1 if p==g else 0 for p,g in zip(preds, labels)]
    # 平均
    return sum(rewards) / len(rewards)
    # 或者总和： return sum(rewards)


def think_token_content_reward(completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    # 提取answer标签之间的内容
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    print("extracted_responses")
    print(extracted_responses)
    print("answer")
    print(answer)

    for text, ans in zip(extracted_responses, answer):
        # 从标准答案字符串中提取汉字序列和对应标签列表
        ref_chars, ref_tags = extract_ref_chars_and_tags(ans)
        # 将正确答案的标签列表中“不是实体”全改成O
        labels = [ "O" if x == "不是实体" else x for x in ref_tags]

        mapping = map_by_char_sequence(text, ref_chars)
        preds = [tag for _, tag in mapping]

        r_tok = token_level_reward(preds, labels)
        
        rewards.append(r_tok)
    return rewards



def extract_entities(labels):
    entities = []
    start = None
    current_type = None

    for i, lab in enumerate(labels):
        if lab != "不是实体":
            if current_type is None:
                # 新实体开始
                current_type = lab
                start = i
            elif lab != current_type:
                # 类型变化，结束上一个实体
                entities.append((current_type, start, i-1))
                current_type = lab
                start = i
        else:
            # 遇到“不是实体”，结束当前实体（如果有）
            if current_type is not None:
                entities.append((current_type, start, i-1))
                current_type = None
                start = None

    # 别忘了末尾未关闭的实体
    if current_type is not None:
        entities.append((current_type, start, len(labels)-1))

    return entities


def entity_level_reward(preds: List[str], labels: List[str]) -> float:
    # 提取实体：(type, start, end)
    true_ents = set(extract_entities(labels))
    pred_ents = set(extract_entities(preds))

    score = 0.0

    # 1) 对每个预测实体打分
    for p_type, p_start, p_end in pred_ents:
        # 找看它是否和某个真实体「边界完全一致」
        match_boundary = [(t_type, t_start, t_end)
                          for t_type, t_start, t_end in true_ents
                          if t_start == p_start and t_end == p_end]
        if match_boundary:
            # 边界对上了，检查类型
            t_type, _, _ = match_boundary[0]
            if t_type == p_type:
                score += 1.0   # 完全正确
            else:
                score -= 0.5  # 边界对，类型错
        else:
            # 边界错，就看它有没有「类型对」的真实体（界定部分重叠或同类型其他位置）
            match_type = [(t_type, t_start, t_end)
                          for t_type, t_start, t_end in true_ents
                          if t_type == p_type]
            if match_type:
                score -= 0.5  # 类型对、边界错
            else:
                score -= 1.0  # 完全误检

    # 2) 真实体漏检处罚
    for t_type, t_start, t_end in true_ents:
        # 如果没有任何预测实体「边界+类型」或「边界+任意类型」命中，就算漏检
        hit = any(
            (p_start == t_start and p_end == t_end) or (p_type == t_type)
            for p_type, p_start, p_end in pred_ents
        )
        if not hit:
            score -= 1.0

    # 3) 归一化：除以真实体数，保证 reward ∈ [负无限, 正无限]，
    #    也可以改成 / len(true_ents) 再加常数偏移变到 [-1,1] 或 [0,1]
    denom = max(1, len(true_ents))
    return score / denom

def think_entity_content_reward(completions, answer, **kwargs):
    """
    对 preds/labels 做实体级打分：
      - 边界 & 类型 都对： +1.0
      - 边界对、类型错：   -0.5
      - 类型对、边界错：   -0.5
      - 真实体漏检：        -1.0
      - 误检实体多余：      -1.0
    最终 reward = 总得分 / 真实实体数
    """
    responses = [completion[0]['content'] for completion in completions]
    # 提取answer标签之间的内容
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    print("extracted_responses")
    print(extracted_responses)
    print("answer")
    print(answer)

    for text, ans in zip(extracted_responses, answer):
        # 从标准答案字符串中提取汉字序列和对应标签列表
        ref_chars, ref_tags = extract_ref_chars_and_tags(ans)
        # 将正确答案的标签列表中“不是实体”全改成O
        labels = [ "O" if x == "不是实体" else x for x in ref_tags]

        mapping = map_by_char_sequence(text, ref_chars)
        preds = [tag for _, tag in mapping]

        r_tok = entity_level_reward(preds, labels)
        
        rewards.append(r_tok)
    return rewards
