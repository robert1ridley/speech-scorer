import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from os import listdir
from os.path import isfile, join
import codecs
from sklearn.model_selection import train_test_split


def get_topic(file_name):
    topics = {'SMK': 1, 'PTJ': 2}
    for topic in topics.keys():
        if topic in file_name:
            return topics[topic]
    return 1000


def get_score(file_name):
    CEFR = {'A2': 1, 'B1': 2, 'B2': 3, 'XX': 4}
    for level in CEFR.keys():
        if level in file_name:
            return CEFR[level]
    return 1000


def build_text_file(essay_groups):
    data = []
    for group in essay_groups:
        with codecs.open(group[0], mode='r', encoding='UTF8', errors='replace') as input_file:
            group_id = str(group[1])
            essay_id = 0
            for line in input_file:
                line = line.strip()
                if line != '':
                    data.append(group_id + '_' + str(essay_id) + '\t' + str(group[2]) + '\t' + line + '\t' + str(group[3]).strip() + '\n')
                    essay_id += 1
    return data


def split_train_dev_test(data):
    TRAINING_FILE = './data/reformed/train.tsv'
    VALIDATION_FILE = './data/reformed/valid.tsv'
    TEST_FILE = './data/reformed/test.tsv'

    train, val_test = train_test_split(data, test_size=0.4, random_state=42)
    test, val = train_test_split(val_test, test_size=0.5, random_state=42)
    train_text = ''
    val_text = ''
    test_text = ''
    for item in train:
        train_text += item
    for i in val:
        val_text += i
    for j in test:
        test_text += j

    new_train_file = open(TRAINING_FILE, 'w')
    new_train_file.write(train_text)
    new_train_file.close()

    new_val_file = open(VALIDATION_FILE, 'w')
    new_val_file.write(val_text)
    new_val_file.close()

    new_test_file = open(TEST_FILE, 'w')
    new_test_file.write(test_text)
    new_test_file.close()


def main():
    main_directory = './data/ICNALE_Spoken_Monologue_2.0_Transcripts/Merged/Plain Text'
    onlyfiles = [f for f in listdir(main_directory) if isfile(join(main_directory, f))]
    essay_groups = []
    essay_group_id = 0
    for filename in onlyfiles:
        score = get_score(filename)
        topic = get_topic(filename)
        essay_groups.append((main_directory + '/' + filename, essay_group_id, topic, score))
        essay_group_id += 1
    full_data = build_text_file(essay_groups)
    split_train_dev_test(full_data)



if __name__ == '__main__':
    main()