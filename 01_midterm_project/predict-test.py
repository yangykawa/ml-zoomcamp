#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-1000'

customer = {'id': 1000,
 'person_age': 30,
 'person_income': 100000,
 'person_home_ownership': 'RENT',
 'person_emp_length': 6.0,
 'loan_intent': 'PERSONAL',
 'loan_grade': 'B',
 'loan_amnt': 100,
 'loan_int_rate': 22.75,
 'loan_percent_income': 0.001,
 'cb_person_default_on_file': 'N',
 'cb_person_cred_hist_length': 5}


response = requests.post(url, json=customer).json()

if response['loan_status'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)