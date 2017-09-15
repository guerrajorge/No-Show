import pandas as pd

dataset = pd.Dataframe()
# dataset keys descriptions
# TIME_TO_APPT = difference in days between ENCOUNTER_APPOINTMENT_DATETIME and ENCOUNTER_APPT_MADE_DATE
# ENCOUNTER_CONTACT_DATE = date when the clinical contact or encounter was created

# fixing variable dtype=datetime
dataset['ENCOUNTER_CONTACT_DATE'] = pd.to_datetime(dataset['ENCOUNTER_CONTACT_DATE'])
dataset['ENCOUNTER_START_DATE'] = pd.to_datetime(dataset['ENCOUNTER_START_DATE'])
dataset['ENCOUNTER_END_DATE'] = pd.to_datetime(dataset['ENCOUNTER_END_DATE'])
dataset['ENCOUNTER_APPOINTMENT_DATETIME'] = pd.to_datetime(dataset['ENCOUNTER_APPOINTMENT_DATETIME'])
dataset['ENCOUNTER_APPT_MADE_DATE'] = pd.to_datetime(dataset['ENCOUNTER_APPT_MADE_DATE'])
dataset['PATIENT_DATE_OF_BIRTH'] = pd.to_datetime(dataset['PATIENT_DATE_OF_BIRTH'])

# # remove for now
#
# tmp = list()
# for key in dataset.keys():
#     if dataset[key].dtype == 'object':
#         tmp.append(key)
#
# tmp = ['ENCOUNTER_CONTACT_DATE', 'ENCOUNTER_CANCELED_DATE', 'ENCOUNTER_START_DATE', 'ENCOUNTER_END_DATE',
#        'ENCOUNTER_TYPE', 'ENCOUNTER_PAYOR_NAME', 'ENCOUNTER_FINANCIAL_CLASS_NAME', 'ENCOUNTER_DEPARTMENT_NAME',
#        'ENCOUNTER_DEPARTMENT_ABBR', 'ENCOUNTER_DEPARTMENT_SPECIALTY', 'ENCOUNTER_PATIENT_CLASS',
#        'ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'ENCOUNTER_CLASS', 'PATIENT_DATE_OF_BIRTH',
#        'PATIENT_DATE_OF_DEATH', 'ENCOUNTER_APPT_MADE_WEEK_DAY', 'PATIENT_NAME', 'PATIENT_STATUS', 'PATIENT_GENDER'
# ,
#        'PATIENT_ID', 'PATIENT_CITY', 'PATIENT_STATE', 'PATIENT_COUNTY', 'PATIENT_COUNTRY', 'PATIENT_ETHNICITY',
#        'PATIENT_RACE', 'PATIENT_RELIGION', 'PATIENT_LANGUAGE', 'PATIENT_ADDRESS_LINE_1', 'PATIENT_ADDRESS_LINE_2',
#        'PRIMARY_CARE_PROVIDER_NAME', 'EPIC_PATIENT_HYPERLINK', 'PROV_TYPE', 'ADULTS_VS_CHILD',
#        'PATIENT_LANGUAGE_GROUP', 'PATIENT_RELIGION_GROUP', 'LOCATION_LEVEL_4', 'DEPARTMENT_TYPE_2', 'TIMETOAPPT',
#        'PEDIATRIC_PATIENT_AGE_GROUPER', 'ENCOUNTER_APPOINTMENT_DATETIME', 'ENCOUNTER_APPT_MADE_DATE']
#
# dataset.drop(tmp, inplace=True, axis=1)

# dataframe used for later debugging and validation
debugging = pd.DataFrame(dataset['ENCOUNTER_APPOINTMENT_STATUS'])

# obtain current date
date = pd.tslib.Timestamp.now()
# calculate the difference between the current date and the last patient's encounter
timedelta = date - dataset['ENCOUNTER_END_DATE']
# insert the days into the x dataset
x['TIME_ENCOUNTER_END_DATE'] = timedelta.dt.days
