import os
import json
import random
import shutil
import pandas as pd

def create_directory_if_necessary(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_json(filepath, towrite):
    with open(filepath + ".json", "w") as outfile: 
        json.dump(towrite, outfile)

dir_path = 'Medley-solos-DB/'

df = pd.read_csv("Medley-solos-DB_metadata.csv")
df.loc[df["subset"] == "training", "subset"] = 'train' # training in train, naming reasons
df.loc[df["subset"] == "validation", "subset"] = 'train' # validation in train

data_train = {'file_name': [], 'label': []}
data_test = {'file_name': [], 'label': []}

print("Loading audios...")
for file in os.listdir(dir_path):
    if os.path.splitext(file)[1] != '.wav':
     continue
    
    uid = file.split('_')[2][:-4]
    split = df[df['uuid4'] == uid]['subset'].iloc[0]
    label = df[df['uuid4'] == uid]['instrument'].iloc[0].replace(" ", "_")

    if split == 'train':
        data_train['file_name'].append(file)
        data_train['label'].append(label)
    else:
        data_test['file_name'].append(file)
        data_test['label'].append(label)

df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)
print("Audios loaded!")

create_directory_if_necessary('../Medley-solos-DB_split')

classes_to_use = [3, 5, 8]
for n_classes in classes_to_use:
    dest_folder = '../Medley-solos-DB_split/Medley-solos-DB_split_' + str(n_classes)
    create_directory_if_necessary(dest_folder)
    create_directory_if_necessary(dest_folder + '/test')
    create_directory_if_necessary(dest_folder + '/train')

    total_labels = list(df_train['label'].unique())
    random_classes = random.sample(total_labels, n_classes)

    filt_df_train = df_train[df_train['label'].isin(random_classes)]
    filt_df_test = df_test[df_test['label'].isin(random_classes)]

    class_labels = list(filt_df_train['label'].unique())
    class_labels_map = dict()
    for i in range(0,len(class_labels)):
        class_labels_map[class_labels[i]]=i+1

    filt_df_train["label"] = filt_df_train['label'].map(class_labels_map)
    filt_df_test["label"] = filt_df_test['label'].map(class_labels_map)

    for _ in list(filt_df_train['label'].unique()):
        create_directory_if_necessary(dest_folder + '/test/' + str(_))
        create_directory_if_necessary(dest_folder + '/train/' + str(_))

    for index, row in filt_df_train.iterrows():
        file_name = row['file_name']
        label = row['label']

        shutil.copy(dir_path + '/' + file_name, dest_folder + '/train/' + str(label) + '/' + file_name)

    for index, row in filt_df_test.iterrows():
        file_name = row['file_name']
        label = row['label']

        shutil.copy(dir_path + '/' + file_name, dest_folder + '/test/' + str(label) + '/' + file_name)

    filt_df_train.to_csv(dest_folder + '/annotations_train.csv', index=False)
    filt_df_test.to_csv(dest_folder + '/annotations_test.csv', index=False)

    inv_class_labels_map = {v: k for k, v in class_labels_map.items()}

    save_json(dest_folder + '/class_labels_map', class_labels_map)
    save_json(dest_folder + '/inv_class_labels_map', inv_class_labels_map)