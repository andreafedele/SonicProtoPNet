import os
import time
import json
import numpy as np

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

def save_json(filepath, towrite):
    with open(filepath + ".json", "w") as outfile: 
        json.dump(towrite, outfile)

for dataset_folder in ['Medley-solos-DB_split/', 'GTZAN_split/', 'esc50_split/']:
    print("******************************************")
    print("******************************************")
    print("DATASET: ", dataset_folder)
    print("******************************************")
    print("******************************************")

    path = '../datasets/' + dataset_folder 

    for folder_split in os.listdir(path):
        if folder_split == '.DS_Store':
            continue

        ts_path = path + folder_split + '/ts/'

        for type in ['cent', 'spect', 'all']:
            npz_to_open = ts_path + type + '.npz'
            print("Folder, type", folder_split, type)

            data = np.load(npz_to_open)
            X_train, y_train, X_test, y_test, filenames_train, filenames_test = data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['filenames_train'], data['filenames_test']

            print("Train shape:", X_train.shape)
            print("Test shape:", X_test.shape)

            if type == 'cent':
                st = time.time() # start time

                knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance='euclidean')
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                cr = classification_report(y_test, y_pred, output_dict=True)
                save_json(npz_to_open[:-4] + '_results', cr)

                et = time.time() # end time
                print('Execution time (ms): ' + str(et - st))
                
                print("Accuracy:", cr['accuracy'])
                print("Macro Accuracy:", cr['macro avg'])
                print("Weighted Accuracy", cr['weighted avg'])
                print("------------------------------------------------")
            else:
                st = time.time() # start time
                
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                cr = classification_report(y_test, y_pred, output_dict=True)
                save_json(npz_to_open[:-4] + '_results', cr)

                et = time.time() # end time
                print('Execution time (ms): ' + str(et - st))
                
                print("Accuracy:", cr['accuracy'])
                print("Macro Accuracy:", cr['macro avg'])
                print("Weighted Accuracy", cr['weighted avg'])
                print("------------------------------------------------")