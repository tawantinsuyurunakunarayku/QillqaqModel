#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import re
import codecs
import fnmatch
import pandas
import tarfile
import unicodedata
import wave

from glob import glob
from os import makedirs, path, remove, rmdir
from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from util.stm import parse_stm_file

has_number = re.compile(r"[0-9]", flags=re.UNICODE)
punct = re.compile(r'[-\[\]"#$%*+/<=>@^_`|~º¡¿ª°…“”.,:;!?{}()]', flags=re.UNICODE)
digit_pat = re.compile(r"\s+(?P<digit>[0-9])\s+", flags=re.UNICODE)

def _download_and_process_data(data_dir):
    print("Downloading Siminchik data set (97GB) into {} if not already present...".format(data_dir))

    _maybe_extract(data_dir, "tempo", "siminchik.tar")

    quz_files = _maybe_split_wav_and_sentences(data_dir, "tempo", "tempo")

    train_files, dev_files, test_files = _split_sets(quz_files)

    train_files.to_csv(os.path.join(data_dir, "quz-train.csv"), index=False, encoding='utf-8')
    dev_files.to_csv(os.path.join(data_dir, "quz-dev.csv"), index=False, encoding='utf-8')
    test_files.to_csv(os.path.join(data_dir, "quz-test.csv"), index=False, encoding='utf-8')

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _parse_transcription(trans_file):
    transcript = codecs.open(trans_file, "r", "utf-8").read()
    return clean_non_ascii(transcript.replace(u'\ufeff',''))

def _maybe_split_wav_and_sentences(data_dir, trans_data, original_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)

    files = []

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.txt"):
            trans_file = os.path.join(root, filename)
            transcript = _parse_transcription(trans_file)

            # Open wav corresponding to transcription file
            wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + ".wav"
            wav_file = os.path.join(source_dir, wav_filename)

            if not os.path.exists(wav_file):
                print("Skipping. does not exist:" + wav_file)
                continue

            wav_filesize = os.path.getsize(wav_file)
            files.append((os.path.abspath(wav_file), wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

def _split_sets(filelist):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return (filelist[train_beg:train_end], filelist[dev_beg:dev_end], filelist[test_beg:test_end])

def clean_non_ascii(text):
    text = text.replace("’".decode("utf-8"), "'") \
        .replace('´'.decode("utf-8"), "'") \
        .replace('`'.decode("utf-8"), "'") \
        .replace(u'\u02bc', "'") \
        .replace(u'\ufeff', "") \
        .replace('”'.decode("utf-8"), ' ') \
        .replace('“'.decode("utf-8"), ' ') \
        .replace('¨'.decode("utf-8"), ' ') \
        .replace('…'.decode("utf-8"), ' ') \
        .replace('ä'.decode("utf-8"), 'a') \
        .replace('ï'.decode("utf-8"), 'i') \
        .replace('&'.decode("utf-8"), 'y') \
        .replace('à'.decode("utf-8"), 'a') \
        .replace('ä'.decode("utf-8"), 'a') \
        .replace('è'.decode("utf-8"), 'e') \
        .replace('ë'.decode("utf-8"), 'e') \
        .replace('ì'.decode("utf-8"), 'i') \
        .replace('ò'.decode("utf-8"), 'o') \
        .replace('ù'.decode("utf-8"), 'u') \
        .replace('á'.decode("utf-8"), 'a') \
        .replace('é'.decode("utf-8"), 'e') \
        .replace('í'.decode("utf-8"), 'i') \
        .replace('ó'.decode("utf-8"), 'o') \
        .replace('ú'.decode("utf-8"), 'u') \
        .replace('ö'.decode("utf-8"), 'o') \
        .replace('ü'.decode("utf-8"), 'u') \
        .replace('ý'.decode("utf-8"), 'y') \
        .replace('–'.decode("utf-8"), ' ') \
        .replace('‼'.decode("utf-8"), ' ') \
        .replace("mp3", "eme pe hiru")
    text = punct.sub(' ', text)
    text = " " + text + " "

    text = text.strip().lower()
    return text

if __name__ == "__main__":
    _download_and_process_data(sys.argv[1])

