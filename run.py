'''from main.trainer.llm_dpo import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/Llama-3.1-8B-Instruct", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/lpc/models/Llama-3.1-8B-Instruct", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, resume_path='./save_model/llama_youku_500_nt2_new_3/ChatGLM_50000', from_pretrained='/home/lpc/models/Llama-3.1-8B-Instruct', loader_name='LLM_DPO', data_path='youku_500_dpo', max_length=3600, batch_size=2, batch_size_eval = 2, task_name='llama_youku_500_dpo_new10')

for i in trainer(num_epochs=100, lr=1e-5, beta=0.1):
    a = i'''

from main.predictor.llm_nt2 import Predictor

pred = Predictor(model_from_pretrained='/home/lpc/models/Llama-3.1-8B-Instruct', resume_path='./save_model/llama_taobao_250_dpo_new2/ChatGLM_125', data_path="taobao_250_nt2")
result = pred(max_length=512, batch_size=20)