import numpy as np
import pandas as pd
import json


symptom_list = [
    'symptom:Anxiety', 'symptom:Asthma', 'symptom:Anosmia', 'symptom:Alcoholism',
    'symptom:Common cold', 'symptom:Cough', 'symptom:Depression', 'symptom:Fatigue',
    'symptom:Fever', 'symptom:Headache', 'symptom:Nausea', 'symptom:Shortness of breath'
]


region_translater = {
    "US-AL": "Alabama",
    "US-AK": "Alaska",
    "US-AZ": "Arizona",
    "US-AR": "Arkansas",
    "US-CA": "California",
    "US-CO": "Colorado",
    "US-CT": "Connecticut",
    "US-DE": "Delaware",
    "US-FL": "Florida",
    "US-GA": "Georgia",
    "US-HI": "Hawaii",
    "US-IA": "Idaho",
    "US-IL": "Illinois",
    "US-IN": "Indiana",
    "US-IA": "Iowa",
    "US-KS": "Kansas",
    "US-KY": "Kentucky",
    "US-LA": "Louisiana",
    "US-ME": "Maine",
    "US-MD": "Maryland",
    "US-MA": "Massachusetts",
    "US-MI": "Michigan",
    "US-MN": "Minnesota",
    "US-MS": "Mississippi",
    "US-MO": "Missouri",
    "US-MT": "Montana",
    "US-NE": "Nebraska",
    "US-NV": "Nevada",
    "US-NH": "New Hampshire",
    "US-NJ": "New Jersey",
    "US-NM": "New Mexico",
    "US-NY": "New York",
    "US-NC": "North Carolina",
    "US-ND": "North Dakota",
    "US-OH": "Ohio",
    "US-OK": "Oklahoma",
    "US-OR": "Oregon",
    "US-PA": "Pennsylvania",
    "US-RI": "Rhode Island",
    "US-SC": "South Carolina",
    "US-SD": "South Dakota",
    "US-TN": "Tennessee",
    "US-TX": "Texas",
    "US-UT": "Utah",
    "US-VT": "Vermont",
    "US-VA": "Virginia",
    "US-WA": "Washington",
    "US-WV": "West Virginia",
    "US-WI": "Wisconsin",
    "US-WY": "Wyoming",
    "US-DC": "District of Columbia"
}

def event_generate(start_year=2017, stop_year=2022, symptom_list=symptom_list, thres=1.02):
    # get all years
    year_list = [str(i) for i in range(start_year, stop_year+1)]

    # get all regions
    path = './data/2017_country_weekly_2017_US_weekly_symptoms_dataset.csv'
    example_data = pd.read_csv(path)
    region_data = example_data['sub_region_1_code'].fillna('US-X').to_numpy()

    region_list = np.unique(region_data).tolist()

    # generate all data dict 
    data_dict = {}

    for year in year_list:
        path = './data/{}_country_weekly_{}_US_weekly_symptoms_dataset.csv'.format(year,year)
        year_data = pd.read_csv(path)
        year_data['sub_region_1_code'] = year_data['sub_region_1_code'].fillna('US-X')

        for region in region_list:
            region_data = year_data.loc[year_data['sub_region_1_code'] == region]
            symptom_data = region_data[symptom_list]

            for symptom in symptom_list:
                try:
                    data_dict[region][symptom] = np.concatenate((data_dict[region][symptom], symptom_data[symptom].to_numpy()))
                except:
                    try:
                        data_dict[region][symptom] = symptom_data[symptom].to_numpy()
                    except:
                        data_dict[region] = {}
                        data_dict[region][symptom] = symptom_data[symptom].to_numpy()

    # generate outbreak events
    event_mark = {}

    for region in data_dict.keys():
        if region not in event_mark.keys():
            event_mark[region] = {}

        for symptom in data_dict[region].keys():
            if symptom not in event_mark[region].keys():
                event_mark[region][symptom] = {}
            symptom_data = data_dict[region][symptom]

            for i,s in enumerate(symptom_data[:-1]):
                if thres * symptom_data[i] <= symptom_data[i+1]:
                    try:
                        event_mark[region][symptom]['all'].append(i+1)
                    except:
                        event_mark[region][symptom]['all'] = [i+1]

                    if int(i/50) == 0:
                        try:
                            event_mark[region][symptom]['2017'].append(i+1)
                        except:
                            event_mark[region][symptom]['2017'] = [i+1]

                    if int(i/50) == 1:
                        try:
                            event_mark[region][symptom]['2018'].append(i+1)
                        except:
                            event_mark[region][symptom]['2018'] = [i+1]
                    if int(i/50) == 2:
                        try:
                            event_mark[region][symptom]['2019'].append(i+1)
                        except:
                            event_mark[region][symptom]['2019'] = [i+1]
                    if int(i/50) == 3:
                        try:
                            event_mark[region][symptom]['2020'].append(i+1)
                        except:
                            event_mark[region][symptom]['2020'] = [i+1]
                    if int(i/50) == 4:
                        try:
                            event_mark[region][symptom]['2021'].append(i+1)
                        except:
                            event_mark[region][symptom]['2021'] = [i+1]

    for region in event_mark.keys():
        temp_l = []
        for symptom in event_mark[region].keys():
            temp_l.append(len(event_mark[region][symptom]))

    return event_mark


def ili_generate(region_translater=region_translater):
    # get state ili data
    path = './data/ILINet_state.csv'
    state_flu_data = pd.read_csv(path, skiprows=1)

    region_flu_data = {}

    for key in region_translater:
        try:
            region_flu_data[key] = state_flu_data.loc[state_flu_data['REGION'] == region_translater[key]]['%UNWEIGHTED ILI'].to_numpy().astype(float)
        except:
            # skip states with imcomplete data
            continue

    # get national ili data
    path = './data/ILINet_national.csv'
    national_flu_data = pd.read_csv(path, skiprows=1)


    region_flu_data['US-X'] = national_flu_data[national_flu_data['REGION'] == 'X']['%UNWEIGHTED ILI'].to_numpy().astype(float)

    return region_flu_data


def save_json():
    event_mark = event_generate()
    region_flu_data = ili_generate()

    with open("./data/event.json", "w") as outfile:
        json.dump(event_mark, outfile)

if __name__ == '__main__':
    save_json()