# run.py中运行程序
## 1.训练
### 1.1 先跑sft(llm_lora.py)
```python
from main.trainer.llm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/glm-4-9b-chat", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/glm-4-9b-chat", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/glm-4-9b-chat', loader_name='LLM_Chat', data_path='weibo_1000_nt2', max_length=3600, batch_size=2, batch_size_eval = 2, task_name='weibo_1000_nt2_new_1')

for i in trainer(num_epochs=100, lr=1e-5):
    a = i
```
`loader_name`可以设置为`LLM_Chat`（中文数据集） 或 `LLM_Chat_EN`（英文数据集） 

### 1.2 再跑dpo(llm_dpo.py)
```python
from main.trainer.llm_dpo import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/glm-4-9b-chat", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/glm-4-9b-chat", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, resume_path='./save_model/taobao_250_nt2_new_1/ChatGLM_12250', from_pretrained='/home/glm-4-9b-chat', loader_name='LLM_DPO', data_path='taobao_250_dpo', max_length=3600, batch_size=2, batch_size_eval = 2, task_name='taobao_250_dpo_chbd_chfn_1:0.5_tj_new1')

for i in trainer(num_epochs=100, lr=1e-5):
    a = i
```
`loader_name`可以设置为`LLM_DPO`（中文数据集） 或 `LLM_DPO_EN`（英文数据集）

## 2.推理
### 2.1 中文(llm_nt2.py)
！！！注意 ctb6 和 ud 数据集得把llm_nt2.py代码中的`不是实体`改为`不是词性`
```python
from main.predictor.llm_nt2 import Predictor

pred = Predictor(model_from_pretrained='/home/glm-4-9b-chat', resume_path='./save_model/youku_1000_nt2_new_1/ChatGLM_26500', data_path="youku_1000_nt2")
result = pred(max_length=512, batch_size=20)
```
### 2.2 英文(llm_nt2_en.py)
```python
from main.predictor.llm_nt2_en import Predictor

pred = Predictor(model_from_pretrained='/home/glm-4-9b-chat', resume_path='./save_model/youku_1000_nt2_new_1/ChatGLM_26500', data_path="youku_1000_nt2")
result = pred(max_length=512, batch_size=20)
```
 


