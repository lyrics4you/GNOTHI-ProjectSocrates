import os
import sys
import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch
import gdown

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LABELS = ['불평/불만',
     '환영/호의',
     '감동/감탄',
     '지긋지긋',
     '고마움',
     '슬픔',
     '화남/분노',
     '존경',
     '기대감',
     '우쭐댐/무시함',
     '안타까움/실망',
     '비장함',
     '의심/불신',
     '뿌듯함',
     '편안/쾌적',
     '신기함/관심',
     '아껴주는',
     '부끄러움',
     '공포/무서움',
     '절망',
     '한심함',
     '역겨움/징그러움',
     '짜증',
     '어이없음',
     '없음',
     '패배/자기혐오',
     '귀찮음',
     '힘듦/지침',
     '즐거움/신남',
     '깨달음',
     '죄책감',
     '증오/혐오',
     '흐뭇함(귀여움/예쁨)',
     '당황/난처',
     '경악',
     '부담/안_내킴',
     '서러움',
     '재미없음',
     '불쌍함/연민',
     '놀람',
     '행복',
     '불안/걱정',
     '기쁨',
     '안심/신뢰']

label_dict = dict(zip(range(len(LABELS)), LABELS))

class EmotionClassifier:
    def __init__(self, label_dict = label_dict, MODEL_WEIGHTS_PATH = ""):
        self.model = load_model()
        if not MODEL_WEIGHTS_PATH:
            MODEL_WEIGHTS_PATH = os.getcwd()
        self.model.load_state_dict(torch.load(get_weights_path(MODEL_WEIGHTS_PATH)))
        self.label_dict = label_dict

    def classify(self, text):
        probs, logits = self.model(text)
        return probs[0].detach().cpu().numpy(), logits[0].detach().cpu().numpy()        

    
    def get_max_n(self, values: list, n = 3):
        max_n_idx = (-values).argsort()[:n]
        max_n_labels, max_n_values = [], []
        for idx in max_n_idx:
            max_n_values.append(values[idx])
            max_n_labels.append(self.label_dict[idx])
        return max_n_labels, max_n_values
    


class load_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)
        
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,   
          return_attention_mask=True,
          return_tensors='pt',
        ).to(device)
        output = self.electra(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:,0,:]
        logits = self.classifier(output)
        probs = torch.sigmoid(logits)
        torch.cuda.empty_cache()
        
        return probs, logits


def get_weights_path(TARGET_PATH):
    REMOTE_PATH = "https://drive.google.com/uc?export=download&id="
    REMOTE_DATA_ID = "1_5n8nPkgLnPbgnsb_iPTyW9NMm_MXpWX"
    REMOTE_URL = REMOTE_PATH + REMOTE_DATA_ID
    LOCAL_PATH = TARGET_PATH
    FILE_NAME = "kote_pytorch_lightning.bin"
    LOCAL_DATA_PATH = os.path.join(LOCAL_PATH, FILE_NAME)
    if FILE_NAME in os.listdir(LOCAL_PATH):
      print("=" * 50)
      print(f'{FILE_NAME} already exists in the directory.')       
      print("=" * 50)
    else:      
      print("=" * 50)
      print('Model weights file does not exists.')
      print("=" * 50)
      gdown.download(REMOTE_URL, LOCAL_DATA_PATH, quiet=False)
    return LOCAL_DATA_PATH


def main():

    
    clf = EmotionClassifier(label_dict, MODEL_WEIGHTS_PATH = "model")
    
    lyrics = """
    멀어져 가는 오후를 바라보다
    스쳐 지나가 버린 그때 생각이나
    기억 모퉁이에 적혀 있던 네가
    지금 여기에 있다
    이젠 멈춰버린 화면 속에서
    내게 여름처럼 웃고 있는 너
    어쩌면 이번이 마지막 goodbye
    오래 머물러 주어서 고마워
    이 말이 뭐라고 그렇게 어려웠을까
    이제 goodbye
    우린 다른 꿈을 찾고 있던 거야
    아주 어린 날 놀던 숨바꼭질처럼
    해가 저물도록 혼자 남은 내가
    지금 여기에 있다
    이미 멈춰버린 화면 속에서
    내게 여름처럼 웃고 있는 너
    어쩌면 이번이 마지막 goodbye
    오래 머물러 주어서 고마워
    이 말이 뭐라고 이렇게 힘들었을까
    Woa- yeah
    You are the only
    You're the only one in my memory, ah
    (You are the all)
    For me
    손에 꼭 쥐었던 너와의 goodbye
    끝내 참지 못한 눈물이 나
    어쩌면 오늘이 마지막 goodbye
    함께 했던 모든 날이 좋았어
    이 말이 뭐라고 그렇게 어려웠을까
    이제 goodbye"""
    
    print(f'Lyrics: \n{lyrics}\n\n')
    
    preds, probs = clf.classify(lyrics, n = 3)
    
    print(f"Predictions: {preds}\nProbability: {probs}")
    
if __name__ == '__main__':
    main()
    
