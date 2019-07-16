#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

from datetime import datetime
from collections import Counter, OrderedDict
import math
import pickle
import pandas as pd
import os
import sys
from data_loader.utils.vocab import Vocab
from tqdm import tqdm
def convert_to_med2vec(patient_data):
    data = []
    for k, vv in patient_data.items():
        for v in vv:
            data.append(v[0])
        data.append([-1])
    return data

if __name__ == '__main__':
    diagnosis_file = sys.argv[1]
    admission_file = sys.argv[2]
    outfile = sys.argv[3]
    med2vec_format = False
    if (len(sys.argv) > 4 and 'med2vec' == sys.argv[4]): #(len(sys.argv) > 4):
        med2vec_format = True
    df_diagnosis = pd.read_csv(diagnosis_file)
    df_admission = pd.read_csv(admission_file)

    full_digit_icd9 = True #flag to extrat short
    # REMOVE 'ORGAN DONOR ACCOUNT' , 'DONOR ACCOUNT' , AND 'ORGAN DONOR' DIAGNOSIS ROWS
    REMOVE_DIAGNOSIS = ~((df_admission['DIAGNOSIS'] == 'ORGAN DONOR ACCOUNT') | (df_admission['DIAGNOSIS'] == 'ORGAN DONOR') | \
                       (df_admission['DIAGNOSIS'] == 'DONOR ACCOUNT'))
    df = df_admission[REMOVE_DIAGNOSIS]

    patient_data = {}
    patient_id = set(df['SUBJECT_ID'])
    vocab = Vocab()
    for pid in tqdm(patient_id):
        pid_df = df[df['SUBJECT_ID'] == pid]
        if (len(pid_df) < 2):
            continue
        adm_list = pid_df[['HADM_ID', 'ADMITTIME', 'DEATHTIME']] # add DISCHATIME ?
        patient_data[pid] = []
        for i, r in adm_list.iterrows():
            admid = r['HADM_ID']
            admitime = r['ADMITTIME']
            icd9_raw = df_diagnosis[df_diagnosis['HADM_ID'] == admid]['ICD9_CODE'].values
            icd9_raw = list(map(str, icd9_raw))
            icd9 = vocab.convert_to_ids(icd9_raw)
            mortality = r['DEATHTIME'] == r['DEATHTIME'] # check not nan
            admtime = datetime.strptime(r['ADMITTIME'], '%Y-%m-%d %H:%M:%S') # TODO: convert date time to integers.. ?!?
            tup = (icd9, admtime, mortality)
            patient_data[pid].append(tup)
    outdir = os.path.abspath(os.path.curdir)
    if not os.path.exists(os.path.join(outdir, 'data')):
        os.mkdir(os.path.join(outdir, 'data'))
    outfile = os.path.join(outdir, 'data', outfile)
    if (med2vec_format):
        patient_data = convert_to_med2vec(patient_data)
    if (med2vec_format):
        pickle.dump(patient_data, open('med2vec.seqs', 'wb'), -1)
        pickle.dump(vocab, open('med2vec.vocab', 'wb'), -1)
    else:
        pickle.dump(patient_data, open(outfile + '_mimic_iii.seqs', 'wb'), -1)
        pickle.dump(vocab, open(outfile + '.vocab', 'wb'), -1)
