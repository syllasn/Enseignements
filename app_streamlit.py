import streamlit as st 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt



st.write("Bienvenue sur notre application")

# open=st.number_input(label="Entrez la valeure d 'ouverture")
# st.write(open)
# high=st.number_input(label="Entrez la grande  valeure ")
# low=st.number_input(label="Entrez la valeure minimale")
# close=st.number_input(label="Entrez la valeure de fermiture")
# volume=st.number_input(label="Entrez le volume")

#Importez des fichiers
file=st.file_uploader("Importez votre fichier")
data=pd.read_csv(file)
#st.write(data.head())

action= st.sidebar.selectbox(
    'Choisir votre action',
    ('AAL', 'NRG', 'NOC')
)

data_AAL= data[data['Name'] == action]
date = pd.to_datetime(data_AAL['date'])
close= data_AAL['close']
st.write(data_AAL.head())
chart_data=pd.DataFrame(date,close)
st.line_chart(chart_data)

### STRIP_START ###
data_AAL=data_AAL.sort_values('date')
data_AAL.head()
#close_value=data_AAL['close'].iloc[0:99]
close_value=data_AAL['close']
data_AAL.head()
X=[]
Y=[]
for i in range( 30, len(close_value)):
    X.append(data_AAL.iloc[i-30:i][['open', 'high', 'low', 'close', 'volume']])
    Y.append(close_value[i])
X = np.array(X)
y = np.array(Y)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_scaled=(X-np.mean(X,axis=0))/np.std(X,axis=0)

#X=StandardScaler().fit_transform(X)

pr_test=st.number_input(label="Donner le pourcentage de train")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=pr_test,  shuffle=False)

#Les donnees
# x=[open,high,low,close,volume]
button=st.button(label='Valider')
if(button): 
    model=pickle.load(open('model_rnn_bourse.sav','rb'))
    pred=model.predict(X_test)
    st.write(pred)
    
    # fig=plt.plot(range(len(y_train), len(y_train) + len(y_test)), pred, label='Predicted (Test)')
    # st.pyplot(fig) 4616

    fig, ax = plt.subplots()
    ax.plot(range(len(y_train), len(y_train) + len(y_test)), pred, label='Predicted (Test)')

    st.pyplot(fig)

#Backend










#Importez des fichiers
# file=st.file_uploader("Importez votre fichier")
# data=pd.read_csv(file)
# st.write(data.head())


#st.button(label='Se connecter')