import csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import vg

def run_tsne():
    file_names = ["index_flex_out.csv","middle_flex_out.csv","ring_flex_out.csv","pinky_flex_out.csv","thumb_flex_out.csv"]
    """while True:
        name = raw_input("Please input a file name to run t-SNE on (or \"quit\" to stop inputting): ")
        if name == "quit":
            break
        file_names.append(name)"""
    
    data = []
    for i in range(len(file_names)):
        with open(file_names[i]) as file:
            reader = list(csv.reader(file))
            data.append([])
            for j in range(len(reader)):
                data[i].append([])
                for k in range(len(reader[j])):
                    data[i][j].append(float(reader[j][k]))

    index_end = len(data[0])
    middle_end = len(data[1]) + index_end
    ring_end = len(data[2]) + middle_end
    pinky_end = len(data[3]) + ring_end

    #Prep and run TSNE
    tsne_data = np.asarray(data[0] + data[1] + data[2] + data[3] + data[4])
    tsne_data = TSNE().fit_transform(tsne_data)
    for j in range(len(tsne_data)):
        with open('TSNE_out.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(tsne_data[j].tolist())

    plt.plot(tsne_data[:index_end,0], tsne_data[:index_end,1], 'ro',
        tsne_data[index_end:middle_end,0], tsne_data[index_end:middle_end,1], 'bo',
        tsne_data[middle_end:ring_end,0], tsne_data[middle_end:ring_end,1], 'go',
        tsne_data[ring_end:pinky_end,0], tsne_data[ring_end:pinky_end,1], 'co',
        tsne_data[pinky_end:,0], tsne_data[pinky_end:,1], 'mo'
        )

    red_patch = mpatches.Patch(color='red', label='index flexion')
    blue_patch = mpatches.Patch(color='blue', label='middle flexion')
    green_patch = mpatches.Patch(color='green', label='ring flexion')
    cyan_patch = mpatches.Patch(color='cyan', label='pinky flexion')
    magenta_patch = mpatches.Patch(color='magenta', label='thumb flexion')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(handles=[red_patch, blue_patch, green_patch, cyan_patch, magenta_patch])
    plt.show()

run_tsne()