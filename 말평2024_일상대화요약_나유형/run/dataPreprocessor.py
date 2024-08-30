import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 사용할 device(GPU)를 설정해주세요.
device = "cuda:6"

model = AutoModelForCausalLM.from_pretrained("maywell/EXAONE-3.0-7.8B-Instruct-Llamafied").to(device)
tokenizer = AutoTokenizer.from_pretrained("maywell/EXAONE-3.0-7.8B-Instruct-Llamafied")

with open('./resource/data/new_train_cutted.json',"r",encoding="utf-8") as f:
    data = json.load(f)

for i in tqdm(range(len(data))):
    chat = data[i]['output']
    print(chat)
    prompt =  "다음 사항을 준수하여 문장을 수정해주세요. 문장에 잘못된 부분, 논리가 어색한 부분이나 표준어가 아닌 부분이 있다면 BLEURT 점수가 높게 나오도록 고쳐주세요. 수정할 부분이 없다면, 원래 문장을 출력해주세요. 수정된 최종 지문의 맨 앞과 맨 뒤에는 [SEP] 토큰을 붙여 전처리하기 편하게 출력해주세요. [예시 1]\n[INPUT] '저는 오늘 럭키한 내 자신을 보았다.' \n[OUTPUT] [SEP]'저는 오늘 운이 좋은 저 자신을 보았습니다.'[SEP]  \n\n[예시 2]\n'나는 오늘 친구들과 밥을 먹었습니다.' \n[OUTPUT] [SEP]'나는 오늘 친구들과 밥을 먹었습니다.'[SEP]  \n\n 위 예시를 참고해서 다음 문장을 고쳐주세요. [SEP]"
    message = [
        {"role" : "system", "content": "You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."},
        {"role" : "user", "content": prompt+chat}
    ]
    source = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    outputs = model.generate(source.to(device),
                            max_new_tokens = 1024,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            use_cache=True,)
    result = tokenizer.decode(outputs[0],skip_special_tokens=True).split("[|assistant|]",1)[-1].strip()
    data[i]['output'] = result
    print(data[i]['output'])
    
for i in range(len(data)):
    curr_output = data[i]['output']
    curr_output.replace("\n"," ")
    if "[SEP]" in curr_output:
        curr_output = curr_output.split("[SEP]")[1]
    else:
        curr_output = curr_output.split("수정")[0]
    if "수정" in curr_output:
        curr_output = curr_output.split("수정")[0]
    if "이 문장은" in curr_output:
        curr_output = curr_output.split("이 문장은")[0]
    data[i]['output'] = curr_output

for i in range(len(data)):
    data[i]['output'] = data[i]['output'].replace("\n\n", "")
    data[i]['output'] = data[i]['output'].replace("\n", "")
    print(data[i]['output'])
    
with open('./resource/data/new_train_cutted_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)