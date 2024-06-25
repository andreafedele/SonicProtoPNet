import os
import json
import random
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# --- UTILS FUNCTIONS -----
def create_directory_if_necessary(path):
    if not os.path.exists(path):
        os.mkdir(path)

def move_audios(df, train_test_path, dataset_dir, split_dir, inv_class_labels_map):
    for __, row in df.iterrows():
        file_name = row['file_name']
        label = row['label']

        create_directory_if_necessary(split_dir + train_test_path + str(label))
        src = dataset_dir + '/' + inv_class_labels_map[label] + '/' + file_name
        dst = split_dir + train_test_path + str(label) + '/' + file_name

        shutil.copy(src, dst)

def save_json(filepath, towrite):
    with open(filepath + ".json", "w") as outfile: 
        json.dump(towrite, outfile)

if __name__ == "__main__":
    # parsing argouments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dir", help="dataset dir")
    argParser.add_argument("-size", help="test percentage size")
    argParser.add_argument("-rs", help="random state")

    args = argParser.parse_args()
    print("args=%s" % args)
    
    DATASET_DIR = args.dir
    SPLIT_DIR = DATASET_DIR + '_split'
    TEST_SIZE = args.size if args.size else 0.25
    RANDOM_STATE = args.rs if args.size else 94

    class_labels = os.listdir(DATASET_DIR)
    if '.DS_Store' in class_labels:
        class_labels.remove('.DS_Store')
   
    # filtering on different classes to be used in each dataset (only 2, 3, 5 or all classes) 
    ## -- THIS IS ONLY USED FOR ESC50 and GTZAN, as MEDLEYS-DB has its own train/test split ---
    classes_to_use = [3, 5, 10, len(class_labels)] if DATASET_DIR == 'esc50' else [3, 5, len(class_labels)]
    classes_to_use = [random.sample(class_labels, n) for n in classes_to_use]

    for class_labels in classes_to_use:
        DEST_DIR = SPLIT_DIR + '/' + SPLIT_DIR + '_' + str(len(class_labels)) 
        
        # creates whole dataframe
        data = {'file_name': [], 'label': []}

        class_labels_map = dict()
        for i in range(0,len(class_labels)):
            class_labels_map[class_labels[i]]=i+1

        for class_label in class_labels:
            audios = os.listdir(DATASET_DIR + '/' + class_label)
            # print(audios)
            for audio in audios:
                data['file_name'].append(audio)
                data['label'].append(class_labels_map[class_label])

        # creates df 
        df = pd.DataFrame(data)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(df['file_name'], df['label'], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label'])
        df_train = pd.DataFrame({'file_name': X_train, 'label':y_train})
        df_test = pd.DataFrame({'file_name': X_test, 'label':y_test})

        create_directory_if_necessary(SPLIT_DIR)
        create_directory_if_necessary(DEST_DIR)
        create_directory_if_necessary(DEST_DIR + '/train')
        create_directory_if_necessary(DEST_DIR + '/test')

        # exporting train/test 
        df_train.to_csv(DEST_DIR + '/annotations_train.csv', index=False)
        df_test.to_csv(DEST_DIR + '/annotations_test.csv', index=False)

        # inverse class labels map
        inv_class_labels_map = {v: k for k, v in class_labels_map.items()}

        # exporting class maps
        save_json(DEST_DIR + '/class_labels_map', class_labels_map)
        save_json(DEST_DIR + '/inv_class_labels_map', inv_class_labels_map)

        # moving audios
        move_audios(df_train, '/train/', DATASET_DIR, DEST_DIR, inv_class_labels_map)
        move_audios(df_test, '/test/', DATASET_DIR, DEST_DIR, inv_class_labels_map)

