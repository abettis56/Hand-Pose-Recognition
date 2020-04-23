import csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def run_tsne():
    file_names = ["index_flex_out.csv","middle_flex_out.csv","ring_flex_out.csv","pinky_flex_out.csv","thumb_flex_out.csv",
                    "index_flexed.csv", "middle_flexed.csv", "ring_flexed.csv", "pinky_flexed.csv", "thumb_flexed.csv"]
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
    thumb_end = len(data[4]) + pinky_end
    ok_end = len(data[5]) + thumb_end
    thumbs_up_end = len(data[6]) + ok_end
    finger_gun_end = len(data[7]) + thumbs_up_end
    nineth_end = len(data[8]) + finger_gun_end


    #Prep and run TSNE and PCA
    formatted_data = np.asarray(data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7] + data[8] + data[9])
    #formatted_data = np.asarray(data[5] + data[6] + data[7] + data[8])

    #Normalize data
    #if(np.amin(formatted_data) < 0):
    #    formatted_data = formatted_data + np.amin(formatted_data)
    #formatted_data = formatted_data / np.amax(formatted_data)

    """tsne_data = TSNE().fit_transform(formatted_data)"""
    pca_data = PCA(n_components=2).fit_transform(formatted_data)
    for j in range(len(formatted_data)):
        with open('PCA_out.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(pca_data[j].tolist())
        """with open('TSNE_out.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(tsne_data[j].tolist())"""

    """#Plot TSNE
    plt.plot(tsne_data[:index_end,0], tsne_data[:index_end,1], 'r.',
        tsne_data[index_end:middle_end,0], tsne_data[index_end:middle_end,1], 'b.',
        tsne_data[middle_end:ring_end,0], tsne_data[middle_end:ring_end,1], 'g.',
        tsne_data[ring_end:pinky_end,0], tsne_data[ring_end:pinky_end,1], 'c.',
        tsne_data[pinky_end:thumb_end,0], tsne_data[pinky_end:thumb_end,1], 'm.',
        tsne_data[thumb_end:ok_end,0], tsne_data[thumb_end:ok_end,1], 'y^',
        tsne_data[ok_end:thumbs_up_end,0], tsne_data[ok_end:thumbs_up_end,1], 'k^',
        tsne_data[thumbs_up_end:finger_gun_end,0], tsne_data[thumbs_up_end:finger_gun_end,1], 'orange',
        tsne_data[finger_gun_end:,0], tsne_data[finger_gun_end:,1], 'pink'
        )"""

    red_patch = mpatches.Patch(color='yellow', label='index flexion')
    blue_patch = mpatches.Patch(color='blue', label='middle flexion')
    green_patch = mpatches.Patch(color='green', label='ring flexion')
    cyan_patch = mpatches.Patch(color='cyan', label='pinky flexion')
    magenta_patch = mpatches.Patch(color='magenta', label='thumb flexion')
    bound = mpatches.Patch(color='orange', label='hypothetical bound')
    arthritis = mpatches.Patch(color='red', label='hypothetical diseased hand')
    ok_pose = mpatches.Patch(color='yellow', label='index_flexed')
    thumbs_up_pose = mpatches.Patch(color='green', label='middle_flexed')
    point_pose = mpatches.Patch(color='yellow', label='ring_flexed')
    fist_pose = mpatches.Patch(color='green', label='pinky_flexed')
    """plt.title("TSNE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(handles=[red_patch, blue_patch, green_patch, cyan_patch, magenta_patch,
                ok_pose, thumbs_up_pose, point_pose, fist_pose])
    plt.show()"""

    #Plot PCA
    ax = plt.subplot(111)
    chartBox = ax.get_position()
    ax.plot(pca_data[:index_end,0], pca_data[:index_end,1], 'y.',
        pca_data[index_end:middle_end,0], pca_data[index_end:middle_end,1], 'g.',
        pca_data[middle_end:ring_end,0], pca_data[middle_end:ring_end,1], 'b.',
        pca_data[ring_end:pinky_end,0], pca_data[ring_end:pinky_end,1], 'c.',
        pca_data[pinky_end:thumb_end,0], pca_data[pinky_end:thumb_end,1], 'm.',
        #60, -40, 'rx',
        pca_data[thumb_end:ok_end,0], pca_data[thumb_end:ok_end,1], 'ko',
        pca_data[ok_end:thumbs_up_end,0], pca_data[ok_end:thumbs_up_end,1], 'ko',
        pca_data[thumbs_up_end:finger_gun_end,0], pca_data[thumbs_up_end:finger_gun_end,1], 'ko',
        pca_data[finger_gun_end:nineth_end,0], pca_data[finger_gun_end:nineth_end,1], 'ko',
        pca_data[nineth_end:,0], pca_data[nineth_end:,1], 'ko'
        )

    plt.title("Finger Flexion with Hypothetical Boundary")
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.7, chartBox.height])
    ax.legend(handles=[red_patch, blue_patch, green_patch, cyan_patch, 
    magenta_patch, bound, arthritis], loc='center left', bbox_to_anchor=(1.00, 0.8))
    plt.show()

run_tsne()