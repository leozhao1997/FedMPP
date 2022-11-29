import json
import numpy as np


client_list = [
    'US-AK', 'US-AL', 'US-AR', 'US-AZ', 'US-CA', 'US-CO', 'US-CT', 
    'US-DC', 'US-DE', 'US-FL', 'US-GA', 'US-HI', 'US-IA', 'US-ID', 
    'US-IL', 'US-IN', 'US-KS', 'US-KY', 'US-LA', 'US-MA', 'US-MD', 
    'US-ME', 'US-MI', 'US-MN', 'US-MO', 'US-MS', 'US-MT', 'US-NC', 
    'US-ND', 'US-NE', 'US-NH', 'US-NJ', 'US-NM', 'US-NV', 'US-NY', 
    'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI', 'US-SC', 'US-SD', 
    'US-TN', 'US-TX', 'US-UT', 'US-VA', 'US-VT', 'US-WA', 'US-WI', 
    'US-WV', 'US-WY', 'US-X'
]


symptom_list = [
    'symptom:Anxiety', 'symptom:Asthma', 
    # 'symptom:Anosmia', 
    'symptom:Alcoholism',
    'symptom:Common cold', 'symptom:Cough', 'symptom:Depression', 'symptom:Fatigue',
    'symptom:Fever', 'symptom:Headache', 'symptom:Nausea', 'symptom:Shortness of breath'
]

with open('./data/event.json') as f:
    all_data = json.load(f)


train_year = [str(i) for i in range(2017,2021)]
test_year = '2021'

client_train_data = []
for client in client_list:
    train_data = []
    max_len = 0
    for year in train_year:
        year_data = None
        for i,symptom in enumerate(symptom_list):
            temp_sym_data = np.array(all_data[client][symptom][year]).reshape(-1,1)
            symptom_code = np.ones_like(temp_sym_data) * i

            temp_sym_data = np.concatenate((temp_sym_data, symptom_code), axis=-1)

            try:
                year_data = np.concatenate((year_data, temp_sym_data), axis=0)
            except:
                year_data = temp_sym_data

        year_data = year_data[np.argsort(year_data[:,0])]
        max_len = max([max_len, len(year_data)])

        train_data.append(year_data)

    for i in range(len(train_data)):
        try:
            train_data[i] = np.concatenate((train_data[i], np.zeros([max_len-len(train_data[i]), 2])), axis=0)
        except:
            continue

    client_train_data.append(train_data)


client_test_data = []

for client in client_list:
    test_data = []
    max_len = 0
    year_data = None
    for i,symptom in enumerate(symptom_list):
        temp_sym_data = np.array(all_data[client][symptom][test_year]).reshape(-1,1)
        symptom_code = np.ones_like(temp_sym_data) * i

        temp_sym_data = np.concatenate((temp_sym_data, symptom_code), axis=-1)

        try:
            year_data = np.concatenate((year_data, temp_sym_data), axis=0)
        except:
            year_data = temp_sym_data

    year_data = year_data[np.argsort(year_data[:,0])]
    max_len = max([max_len, len(year_data)])

    test_data.append(year_data)

    for i in range(len(test_data)):
        try:
            test_data[i] = np.concatenate((test_data[i], np.zeros([max_len-len(test_data[i]), 2])), axis=0)
        except:
            continue

    client_test_data.append(test_data)




np.save('./train.npy', client_train_data)
np.save('./test.npy', client_test_data)

client_train_data = np.array(client_train_data)
client_test_data = np.array(client_test_data)

print(client_test_data[2,0].shape)