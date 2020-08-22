"""
This program is designed to act as a sort of middleman between recorded data and visualization.

It accepts a 135-dimensional vector describing a hand, with each bone described relative to the previous.
Then, the vector is converted to describe each bone relative to a common origin point.

If there's an issue with displaying data visually, it's probably here or in "index.html"
"""

import csv
import numpy as np

def read():
    #Get and prep data
    file_name = "hand_manifold.csv"
    
    data = []
    with open(file_name) as file:
        reader = list(csv.reader(file))
        for i in range(len(reader)):
            data.append([])
            for j in range(len(reader[i])):
                data[i].append(float(reader[i][j]))

    return data

def convert():
    #currently just set to read the 1st vector found in the .csv file.
    #Once shown to be in complete working order, would read and convert all vectors
    vector = read()[0]

    vector = np.asarray(vector)

    vector = vector.reshape((-1, 3))

    #each finger works the same way, attempting to undo the normalization procedure from basis_recorder.py

    ####################rotate thumb#######################
    distal_matrix = np.concatenate((vector[6].reshape((-1, 1)),
                                vector[7].reshape((-1, 1)),
                                vector[8].reshape((-1, 1))), axis=1)
    intermediate_matrix = np.concatenate((vector[3].reshape((-1, 1)),
                                vector[4].reshape((-1, 1)),
                                vector[5].reshape((-1, 1))), axis=1)
    proximal_matrix = np.concatenate((vector[0].reshape((-1, 1)),
                                vector[1].reshape((-1, 1)),
                                vector[2].reshape((-1, 1))), axis=1)

    for i in range(6, 9):
        vector[i] = np.linalg.solve(distal_matrix, vector[i]).T

    for i in range(3, 6):
        vector[i] = np.linalg.solve(intermediate_matrix, vector[i]).T
        vector[i+3] = np.linalg.solve(intermediate_matrix, vector[i]).T

    for i in range(0, 3):
        vector[i+3] = np.linalg.solve(proximal_matrix, vector[i]).T
        vector[i+6] = np.linalg.solve(proximal_matrix, vector[i]).T

    ####################rotate index#######################
    distal_matrix = np.concatenate((vector[15].reshape((-1, 1)),
                                vector[16].reshape((-1, 1)),
                                vector[17].reshape((-1, 1))), axis=1)
    intermediate_matrix = np.concatenate((vector[12].reshape((-1, 1)),
                                vector[13].reshape((-1, 1)),
                                vector[14].reshape((-1, 1))), axis=1)
    proximal_matrix = np.concatenate((vector[9].reshape((-1, 1)),
                                vector[10].reshape((-1, 1)),
                                vector[11].reshape((-1, 1))), axis=1)

    for i in range(15, 18):
        vector[i] = np.linalg.solve(distal_matrix, vector[i]).T

    for i in range(12, 15):
        vector[i] = np.linalg.solve(intermediate_matrix, vector[i]).T
        vector[i+3] = np.linalg.solve(intermediate_matrix, vector[i+3]).T

    for i in range(9, 12):
        vector[i+3] = np.linalg.solve(proximal_matrix, vector[i+3]).T
        vector[i+6] = np.linalg.solve(proximal_matrix, vector[i+6]).T

    ####################rotate middle#######################
    distal_matrix = np.concatenate((vector[24].reshape((-1, 1)),
                                vector[25].reshape((-1, 1)),
                                vector[26].reshape((-1, 1))), axis=1)
    intermediate_matrix = np.concatenate((vector[21].reshape((-1, 1)),
                                vector[22].reshape((-1, 1)),
                                vector[23].reshape((-1, 1))), axis=1)
    proximal_matrix = np.concatenate((vector[18].reshape((-1, 1)),
                                vector[19].reshape((-1, 1)),
                                vector[20].reshape((-1, 1))), axis=1)

    for i in range(24, 27):
        vector[i] = np.linalg.solve(distal_matrix, vector[i]).T

    for i in range(21, 24):
        vector[i] = np.linalg.solve(intermediate_matrix, vector[i]).T
        vector[i+3] = np.linalg.solve(intermediate_matrix, vector[i]).T

    for i in range(18, 21):
        vector[i+3] = np.linalg.solve(proximal_matrix, vector[i+3]).T
        vector[i+6] = np.linalg.solve(proximal_matrix, vector[i+6]).T

    ####################rotate ring#######################
    distal_matrix = np.concatenate((vector[33].reshape((-1, 1)),
                                vector[34].reshape((-1, 1)),
                                vector[35].reshape((-1, 1))), axis=1)
    intermediate_matrix = np.concatenate((vector[30].reshape((-1, 1)),
                                vector[31].reshape((-1, 1)),
                                vector[32].reshape((-1, 1))), axis=1)
    proximal_matrix = np.concatenate((vector[27].reshape((-1, 1)),
                                vector[28].reshape((-1, 1)),
                                vector[29].reshape((-1, 1))), axis=1)

    for i in range(33, 36):
        vector[i] = np.linalg.solve(distal_matrix, vector[i]).T

    for i in range(30, 33):
        vector[i] = np.linalg.solve(intermediate_matrix, vector[i]).T
        vector[i+3] = np.linalg.solve(intermediate_matrix, vector[i]).T

    for i in range(27, 30):
        vector[i+3] = np.linalg.solve(proximal_matrix, vector[i+3]).T
        vector[i+6] = np.linalg.solve(proximal_matrix, vector[i+6]).T

    ####################rotate pinky#######################
    distal_matrix = np.concatenate((vector[42].reshape((-1, 1)),
                                vector[43].reshape((-1, 1)),
                                vector[44].reshape((-1, 1))), axis=1)
    intermediate_matrix = np.concatenate((vector[39].reshape((-1, 1)),
                                vector[40].reshape((-1, 1)),
                                vector[41].reshape((-1, 1))), axis=1)
    proximal_matrix = np.concatenate((vector[36].reshape((-1, 1)),
                                vector[37].reshape((-1, 1)),
                                vector[38].reshape((-1, 1))), axis=1)

    for i in range(42, 45):
        vector[i] = np.linalg.solve(distal_matrix, vector[i]).T

    for i in range(39, 42):
        vector[i] = np.linalg.solve(intermediate_matrix, vector[i]).T
        vector[i+3] = np.linalg.solve(intermediate_matrix, vector[i]).T

    for i in range(36, 39):
        vector[i+3] = np.linalg.solve(proximal_matrix, vector[i+3]).T
        vector[i+6] = np.linalg.solve(proximal_matrix, vector[i+6]).T
    """
    vector[15] = np.linalg.solve(distal_matrix, vector[15]).T
    vector[16] = np.linalg.solve(distal_matrix, vector[16]).T
    vector[17] = np.linalg.solve(distal_matrix, vector[17]).T
    
    vector[12] = np.linalg.solve(intermediate_matrix, vector[12]).T
    vector[13] = np.linalg.solve(intermediate_matrix, vector[13]).T
    vector[14] = np.linalg.solve(intermediate_matrix, vector[14]).T
    vector[15] = np.linalg.solve(intermediate_matrix, vector[12]).T
    vector[16] = np.linalg.solve(intermediate_matrix, vector[13]).T
    vector[17] = np.linalg.solve(intermediate_matrix, vector[14]).T
    """

    output(vector.flatten().tolist())

def output(vector):
    vector = [vector]
    with open('displayable.csv', 'ab') as file:
        writer = csv.writer(file)
        writer.writerow(vector[0])
        file.close()
    
convert()



"""
    #normalize rotation of distal bone relative to intermediate bone
    matrix = np.concatenate((vector[3].reshape((-1, 1)),
                                vector[4].reshape((-1, 1)),
                                vector[5].reshape((-1, 1))), axis=1)

    distal_bone_x = np.linalg.solve(vector[], matrix).T
    distal_bone_y = np.linalg.solve(vector[], matrix).T
    distal_bone_z = np.linalg.solve(vector[], matrix).T
"""