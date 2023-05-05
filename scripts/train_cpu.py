#!/usr/bin/env python

import os
import subprocess

CoNLL2000_DIR = CoNLL2000_DIR = "../data/CoNLL2000"

os.makedirs('ms_log', exist_ok=True)
CUR_DIR = os.getcwd()
os.environ['GLOG_log_dir'] = os.path.join(CUR_DIR, 'ms_log')
os.environ['GLOG_logtostderr'] = '0'

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_PATH, '..', 'default_config.yaml')

    subprocess.Popen(['python', '../train.py',
                      '--config_path=' + CONFIG_FILE,
                      '--device_target=CPU',
                      '--data_CoNLL_path=' + CoNLL2000_DIR,
                      '--build_data=False',
                      '--preprocess=true',
                      '--preprocess_path=./preprocess'],
                     stdout=open('log_train_cpu.txt', 'w'),
                     stderr=subprocess.STDOUT)
