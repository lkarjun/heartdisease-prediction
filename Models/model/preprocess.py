from sklearn.base import TransformerMixin


REPLACE_VALUES = {
                  'Smoking': {'No':0,'Yes':1}, 
                  'AlcoholDrinking': {'No':0,'Yes':1}, 
                  'Stroke': {'No':0,'Yes':1},
                  'DiffWalking': {'No':0,'Yes':1},
                  'Diabetic': {'No':0,'No, borderline diabetes':0,
                               'Yes (during pregnancy)':1,'Yes':1},
                  'PhysicalActivity': {'No':0,'Yes':1},
                  'Asthma': {'No':0,'Yes':1},
                  'KidneyDisease': {'No':0,'Yes':1},
                  'SkinCancer': {'No':0,'Yes':1},
                  'Sex': {'Male':0,'Female':1},
                  'AgeCategory': {'18-24':0,'25-29':1,'30-34':2,'35-39':3,
                                  '40-44':4,'45-49':5,'50-54':6,'55-59':7,
                                  '60-64':8,'65-69':9,'70-74':10,
                                  '75-79':11,'80 or older':12},
                  'GenHealth': {'Excellent':0,'Very good':1,
                                'Good':2,'Fair':3,'Poor':4},
                  'HeartDisease': {'No':0,'Yes':1}
                }


class PreProcess(TransformerMixin):

    def __init__(self, target = 'HeartDisease'):
        self.target = target
    
    def fit(self, X, y = None):
        return self
        
    def transform(self, X, y = None):
        
        for key, value in REPLACE_VALUES.items():
            if key != self.target:
                X[key].replace(value, inplace = True)
        
        if y is not None:
            y.replace(REPLACE_VALUES[self.target], inplace = True)
            return X, y
        return X
