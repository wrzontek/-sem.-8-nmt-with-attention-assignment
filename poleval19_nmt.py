"""Implementation based on https://github.com/pcyin/pytorch_nmt and 
Stanford CS224 2019 class.
"""
import csv
import os

from datasets import load_dataset

_PATH = "en_pl_data/"
os.mkdir("en_pl_data")

# Create data 
dataset = load_dataset("poleval2019_mt", "en-pl")
len_data = len(dataset["train"]) - 1_000

with open(_PATH + "train.en", 'w') as f:
    writer = csv.writer(f)
    for index in range(len_data):
        writer.writerow([dataset["train"][index]["translation"]["en"]])

with open(_PATH + "train.pl", 'w') as f:
    writer = csv.writer(f)
    for index in range(len_data):
        writer.writerow([dataset["train"][index]["translation"]["pl"]])

with open(_PATH + "dev.en", 'w') as f:
    writer = csv.writer(f)
    for index in range(len_data, len_data + 1_000):
        writer.writerow([dataset["train"][index]["translation"]["en"]])

with open(_PATH + "dev.pl", 'w') as f:
    writer = csv.writer(f)
    for index in range(len_data, len_data + 1_000):
        writer.writerow([dataset["train"][index]["translation"]["pl"]])

with open(_PATH + "test.en", 'w') as f:
    writer = csv.writer(f)
    for index, row in enumerate(dataset["validation"]):
        writer.writerow([row["translation"]["en"]])

with open(_PATH + "test.pl", 'w') as f:
    writer = csv.writer(f)
    for index, row in enumerate(dataset["validation"]):
        writer.writerow([row["translation"]["pl"]])
