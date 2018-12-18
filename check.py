#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import re
import fnmatch
import pdb
import codecs
from util import text

alphabet = text.Alphabet("/home/ubuntu/datasets/quz_alphabet.txt")

has_number = re.compile(r"[0-9]", flags=re.UNICODE)
punct = re.compile(r'[-\[\]"#$%*+/<=>@^_`|~º¡¿ª°…“”.,:;!?{}()]', flags=re.UNICODE)
digit_pat = re.compile(r"\s+(?P<digit>[0-9])\s+", flags=re.UNICODE)

def clean_non_ascii(text):
    text = text.replace("’".decode("utf-8"), "'") \
        .replace(u'\ufeff','') \
        .replace('´'.decode("utf-8"), "'") \
        .replace('`'.decode("utf-8"), "'") \
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

for root, dirnames, filenames in os.walk("/home/ubuntu/datasets/tempo/"):
    for filename in fnmatch.filter(filenames, "*.txt"):
        trans_file = os.path.join(root, filename)
        try:
            text.text_to_char_array(clean_non_ascii(codecs.open(trans_file, 'r', 'utf-8').read()), alphabet)
        except Exception as err:
            print(err,)
            print(trans_file)   
            #import pdb; pdb.set_trace()
            continue

