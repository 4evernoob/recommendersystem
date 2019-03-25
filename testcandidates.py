import pandas as pd
import gensim as gs
import warnings
import numpy as np
import math
import json
import unidecode
import requests
#show data from candidate
def listdata(df_candidates, index):
    for expe in df_candidates.loc[df_candidates.id==index,:].values:
     print(list(expe)[4])
     print(list(expe)[3])

#similarity calculation
def cosim(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    if math.isnan(cos):
        ##vector is empty
        return -1
    else:
        return cos

# average vector for sentence
def avvec(model,sentence):
    l=[]
    for i in sentence:
        try:
          l.append(model[i])
        except:
            pass
           # print('{0} skipped shomehow'.format(i))
    if len(l)==0:
        ##none of the words were found so empty vector
        return  [0] * 100
    else:
        #compute vector
        k=sum(l)/len(l)
        return k
#query de descripcion
description='ejecucion pruebas definicion de matrices de prueba ciclos de prueba pruebas dinamicas pruebas de carga'
#'administracion de recursos humanos seleccion de personal '\

description= description.lower().split()
#'cableado estructurado redes inalambricas mantenimiento de servidores deteccion de fallas'.lower().split()
#'implementacion de webservices usando frameworks como Vue react mediante java o php uso de bases nosql mongodb patron de diseño mvc'.lower().split()
#.lower().split()

#query de skills

skills='java selenium testing QA mantis'

skills=skills.lower().split()
#'linux bash administracion de redes configuracion de redes'.lower().split()
#'diseño de software bases de datos no relacionales bases de datos relacionales'.lower().split()
#'java selenium testing QA mantis'.lower().split()
#query de salario actual
sueldoactual=['15,000 a 20,000','10,000 a 15,000']
#region seleccionada
reg=9
#lenguaje not implemented
leng='Inglés'
#configuration
token= 'DDYJc2c725'
#number of samples
number= 400
##extract data
r= requests.get('https://empleosti.com.mx/api/v2/candidates?token='+token+'&take='+str(number)+'&region_id='+str(reg))
#check response code
print(r.status_code)

myr=r.json()#json_normalize(r.json())
#json to pandas dataframe =)
df_candidates =pd.DataFrame.from_records(myr)
#get some details
print(df_candidates.head())
print(list(df_candidates))
#set index in dataframe
df_candidates.set_index('id')

##extract text in description
allxp=''
for i in df_candidates.values:
    #allxp=allxp+" "+
# i
    for j in i[3]:
        allxp=allxp+" "+j['description']
#print(allxp)

#extract all data in skills
allski=''
for i in df_candidates.values:
    for j in i[8]:
        allski=allski+" "+j['name']
#print(allski)

#string cleaning (unfortunately i dont have any custom method so i use default method)
dict1=gs.utils.simple_preprocess(allxp)
dict1=[dict1]
dict2=gs.utils.simple_preprocess(allski)
dict2=[dict2]

#build models from the words
modeldes=gs.models.Word2Vec(dict1,size=100,
window=10,
min_count=1,
workers=10)
modelski=gs.models.Word2Vec(dict2,size=100,
window=4,
min_count=1,
workers=10)
#train models
modeldes.train(dict1, total_examples=len(dict1), epochs=300)
modelski.train(dict2, total_examples=len(dict2), epochs=300)
#query my information
idx=[]
dis1=[]

for i in df_candidates.values:
    if(i[1]in sueldoactual and i[6]== reg and i[5]):
        sentencedes=''
        for j in i[3]:
            sentencedes=sentencedes+" "+j['description']

        sentencedes=sentencedes.lower().split()
        sentenceskil=''
        for j in i[8]:
            sntenceskill=sentenceskil+" "+j['name']


        sentenceskil=sentenceskil.lower().split()

        sentenceqde=description
        sentenceqsk=skills

        if sentencedes!=sentenceqde:
            r1=cosim(avvec(modeldes,sentencedes),avvec(modeldes,sentenceqde))
            r2=cosim(avvec(modelski,sentenceskil),avvec(modelski,sentenceqsk))
            idx.append(i[4])
            #weights
            dis1.append(.75*r1+.25*r2)

          #  print('Comparing {0} with {1} distance {2}'.format(sentencedes,sentenceqde,(.75*r1+.25*r2)))
        else:
            print('value skipped')
    else:
        continue

#ordering my results
ordn=sorted(
                list(
                    zip(
                        idx,
                        dis1
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
print('resultados de acuerdo al analisis de similaridad con {0}'.format(description,skills))
#print best 20 candidates
for i in range(20):

    print('score: {0}'.format(ordn[i][1]))
    print(listdata(df_candidates,ordn[i][0]))
    print('-------------------------')



