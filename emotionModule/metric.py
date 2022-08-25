import pandas as pd
import numpy as np
from p_tqdm import t_map
from tqdm import tqdm
from scipy.special import logit, expit


valence_dict = dict(zip(['불평/불만',
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
 '안심/신뢰'], [-1, 1, 1, -1, 1, -1, -1, 1, 1,
       0, -1, 0, -1, 1, 1, 1, 1,
       -1, -1, -1, -1, -1, -1, -1, 
       -1, -1, -1, 1, 0, -1, -1,
       1, -1, -1, -1, -1, -1, -1, 0,
       1, -1, 1, 1]))



class emotionMetrics:
    def __init__(self, clf):
        self.valence_dict = valence_dict
        self.gid_dict = {"kamjarr": "1zbTIxj0pkhdJX1nNtNK1sMYgdnT_NyNJ",
                         "myungon": "1KggV9naD1-JomqD0ocQQ19tBV-5ROeD5",
                         "yyyybbom": "1hnAZlqlA-yj6h1r6gA7IUEw6FsVsUVkQ",
                         "1004ykw": "1KqUnvU9IrrEf8lhsThpVP3a-Cuyzyv0B"}
        self.clf = clf
        
        
    def get_metrics(self, window_size = 5, stride = 1):
        df_logits = self._get_logits(window_size, stride)    
        df_weights = self._get_weight()        
        df_intensity, intensity, rel_intensity = self._calc_intensity(df_logits, df_weights) 

        valence = self._calc_valence(df_intensity)        
        score = self._calc_score(valence)        
        df_summary = self._get_summary(df_logits, intensity, rel_intensity, valence, score)        

        self.logits = df_logits
        self.intensity = intensity
        self.rel_intensity = rel_intensity
        self.valence = valence
        self.score = score        
        self.summary = df_summary    
    
    def load_data(self, WRITER_NAME):
        REMOTE_PATH = "https://drive.google.com/uc?export=download&id="
        REMOTE_DATA_ID = self.gid_dict[WRITER_NAME]
        REMOTE_URL = REMOTE_PATH + REMOTE_DATA_ID
        raw = pd.read_csv(REMOTE_URL)
        raw.head()
        df = raw.copy().dropna().reset_index(drop = True)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
        self.df = df
        print(f'{WRITER_NAME}\'s data is loaded.')
    
        
    def _get_logits(self, window_size, stride):
        emotion = t_map(lambda x : self._get_emo_pred_df(text = x, window_size = window_size, stride = stride), self.df["text"])
        df_pred = self.df.join(pd.concat(emotion).reset_index(drop = True)).drop(columns = '없음')
        return df_pred.set_index("date").loc[:, list(self.valence_dict.keys())].sort_index().T
    
        
    def _get_weight(self):
        emotion_rate = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vRkS_4d3CuvBEAw-ufZrsDhAiOi6s8E5P8nzS4oOQZbf2wogtNeoNsxq-GmHKQ6B83q1RTx3T6XEesf/pub?gid=1066653645&single=true&output=csv")
        df_emo = pd.DataFrame(self.valence_dict.keys(), columns = ['']).set_index('').join(
            emotion_rate.mean().to_frame(name = 'rating')).dropna()
        df_emo['weight'] = logit(df_emo['rating']/5)
        return df_emo.join(pd.DataFrame.from_dict(valence_dict, orient = 'index', columns = ['valence'])).drop(columns = 'rating')
       
        
    def _calc_intensity(self, df_logits, df_weights):
        df_intensity = df_weights.copy()
        for date, logits in df_logits.iteritems():
            df_intensity[date] = expit(logits + df_weights['weight'])
            
        intensity = df_intensity.mean().drop(index = ['weight', 'valence']).to_frame('intensity') * 100
        rel_intensity = (intensity['intensity'] / intensity['intensity'].mean()).to_frame('rel_intensity')
        
        return df_intensity, intensity, rel_intensity
    
    
    def _calc_valence(self, df_intensity):
        return df_intensity.groupby('valence').mean().drop(columns = ['weight'], index = [0]).T.rename(columns = {-1: 'negative', 1: 'positive'}) * 100
        
        
    def _calc_score(self, valence):
        return (-50 + valence['positive']/ (valence['positive'] + valence['negative']) * 100).to_frame("emotion_score")
    
    
    def _get_summary(self, df_logits, intensity, rel_intensity, valence, overall):
        df_summary = self.df[['date', 'logNo']].set_index('date').sort_index()
        temp = []
        for date, row in df_logits.T.iterrows():
            temp.append(row.nlargest(5).index.tolist())
        df_summary['emotions'] = temp
        return df_summary.join(overall).join(intensity).join(rel_intensity).join(valence).reset_index(drop = True)
           
        
    def _get_emo_pred_df(self, text, logit = True, window_size = 5, stride = 1):
        result = []
        text_split = [''] * (window_size-1) + text.split('\n') + [''] * (window_size -1)
        for idx in range(len(text_split)- window_size + 1):        
            combined_text = ' '.join(text_split[idx+ (stride-1)*idx:idx + window_size + (stride-1)*idx])
            if combined_text:
                result.append(pd.DataFrame([self.clf.classify(combined_text)[int(logit)]], columns = self.clf.label_dict.values()))
        return pd.concat(result).mean().to_frame().T       