"""
This program is responsible for connecting to the Leap Motion Sensor,
detecting hands, normalizing the rotation of the hand using the palm basis vectors
as a base, and saving the basis vectors of each finger bone in the new space to a .csv file.
"""

# This code modified from https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
# 
# Originally written by the Leapmotion devs as a "Hello World" for their system.

from os import name, system
from time import sleep

import os, sys, inspect, thread, time
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'
# Mac
#arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap, keyboard, csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import vg

# define our clear function 
def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear')

#Used to 
def normalize(vector, normal, colinear, cross):
    #convert everything to numpy array
    vector = np.asarray(vector)
    normal = np.asarray(normal)
    colinear = np.asarray(colinear)
    cross = np.asarray(cross)
    #reshape normal, colinear, and cross vectors for change of basis
    ###NOTE:TESTING
    normal = normal * -1
    ###END TESTING
    normal = normal.reshape((-1, 1))
    colinear = colinear.reshape((-1, 1))
    cross = cross.reshape((-1, 1))
    matrix = np.concatenate((cross, normal, colinear), axis=1)

    #reshape vector
    vector = vector.reshape((9, 3))

    #normalize rotation of bones relative to hand position
    proximal_bone_x = np.linalg.solve(matrix, vector[0]).T
    proximal_bone_y = np.linalg.solve(matrix, vector[1]).T
    proximal_bone_z = np.linalg.solve(matrix, vector[2]).T
    for i in range(0, 9):
        vector[i] = np.linalg.solve(matrix, vector[i]).T
    #normalize rotation of intermediate bone relative to proximal bone
    matrix = np.concatenate((vector[0].reshape((-1, 1)),
                                vector[1].reshape((-1, 1)),
                                vector[2].reshape((-1, 1))), axis=1)
    intermediate_bone_x = np.linalg.solve(matrix, vector[3]).T
    intermediate_bone_y = np.linalg.solve(matrix, vector[4]).T
    intermediate_bone_z = np.linalg.solve(matrix, vector[5]).T
    for i in range(3, 9):
        vector[i] = np.linalg.solve(matrix, vector[i]).T
    #normalize rotation of distal bone relative to intermediate bone
    matrix = np.concatenate((vector[3].reshape((-1, 1)),
                                vector[4].reshape((-1, 1)),
                                vector[5].reshape((-1, 1))), axis=1)

    distal_bone_x = np.linalg.solve(matrix, vector[6]).T
    distal_bone_y = np.linalg.solve(matrix, vector[7]).T
    distal_bone_z = np.linalg.solve(matrix, vector[8]).T

    vector = np.vstack((proximal_bone_x, proximal_bone_y, proximal_bone_z,
            intermediate_bone_x, intermediate_bone_y, intermediate_bone_z,
            distal_bone_x, distal_bone_y, distal_bone_z))

    #convert back to list and return
    vector = vector.flatten().tolist()
    return vector

def get_bone_vec(finger):
    #Get bone basis vectors
    vec_set = []
    #get vector for proximal phalanx
    vec = finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    vec_set += [vec.x_basis.x, vec.x_basis.y, vec.x_basis.z,
                vec.y_basis.x, vec.y_basis.y, vec.y_basis.z,
                vec.z_basis.x, vec.z_basis.y, vec.z_basis.z]
    #get vector for intermediate phalanx
    vec = finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    vec_set += [vec.x_basis.x, vec.x_basis.y, vec.x_basis.z,
                vec.y_basis.x, vec.y_basis.y, vec.y_basis.z,
                vec.z_basis.x, vec.z_basis.y, vec.z_basis.z]
    
    #get vector for distal phalanx
    vec = finger.bone(Leap.Bone.TYPE_DISTAL).basis
    vec_set += [vec.x_basis.x, vec.x_basis.y, vec.x_basis.z,
                vec.y_basis.x, vec.y_basis.y, vec.y_basis.z,
                vec.z_basis.x, vec.z_basis.y, vec.z_basis.z]

    return vec_set

def get_vec(vector):
    return [vector.x, vector.y, vector.z]
    

def get_hand_data(hand):
    #Get fingers
    thumb_finger_list = hand.fingers.finger_type(Leap.Finger.TYPE_THUMB)
    thumb = thumb_finger_list[0]

    index_finger_list = hand.fingers.finger_type(Leap.Finger.TYPE_INDEX)
    index = index_finger_list[0]

    middle_finger_list = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)
    middle = middle_finger_list[0]

    ring_finger_list = hand.fingers.finger_type(Leap.Finger.TYPE_RING)
    ring = ring_finger_list[0]

    pinky_finger_list = hand.fingers.finger_type(Leap.Finger.TYPE_PINKY)
    pinky = pinky_finger_list[0]

    #Get vectors
    #Start with finger bone basis vectors
    thumb_vec = get_bone_vec(thumb)
    index_vec = get_bone_vec(index)
    middle_vec = get_bone_vec(middle)
    ring_vec = get_bone_vec(ring)
    pinky_vec = get_bone_vec(pinky)
    #Get hand basis vectors
    """normal = np.asarray([hand.palm_normal.x, hand.palm_normal.y, hand.palm_normal.z])
    normal = np.negative(normal)
    colinear = np.asarray([hand.direction.x, hand.direction.y, hand.direction.z])
    colinear = np.negative(colinear)
    cross = np.cross(normal, colinear)"""
    cross = get_vec(hand.basis.x_basis)
    normal = get_vec(hand.basis.y_basis)
    colinear = get_vec(hand.basis.z_basis)

    #Send basis vectors through normalization
    thumb_vec = normalize(thumb_vec, normal, colinear, cross)
    index_vec = normalize(index_vec, normal, colinear, cross)
    middle_vec = normalize(middle_vec, normal, colinear, cross)
    ring_vec = normalize(ring_vec, normal, colinear, cross)
    pinky_vec = normalize(pinky_vec, normal, colinear, cross)

    row = []
    row += thumb_vec
    row += index_vec
    row += middle_vec
    row += ring_vec
    row += pinky_vec

    print(len(row))

    return row

recording = False
unlocked = True
data = []

class SampleListener(Leap.Listener):
    def on_connect(self, controller):
        print ("Connected")

    def on_frame(self, controller):
        global unlocked
        global data
        global recording

        frame = controller.frame()
        if frame.is_valid:
            hands = frame.hands
            handL = None
            handR = None
            for i in range(0,2):
                if hands[i].is_valid and hands[i].is_right:
                    print ("RIGHT HAND FOUND")
                    #get row data and append it
                    row = get_hand_data(hands[i])
                    if None not in row and recording:
                        data.append(row)
                            
        if keyboard.is_pressed('p') and unlocked:
            unlocked = False
            print("Saving...")

            #Add rows to .csv files
            for i in range(len(data)):
                with open('hand_manifold.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[i])

        if keyboard.is_pressed('q') and unlocked:
            unlocked = False
            if not recording:
                print("Now recording data")
                recording = True
            else:
                print("recording done")
                recording = False

        if (not keyboard.is_pressed('q') and
        not keyboard.is_pressed('p') and
        not unlocked):
            unlocked = True

        #clear()

def main():
    controller = Leap.Controller()
    listener = SampleListener()
    
    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    
    recording = False
    unlocked = True
    
    # Keep this process running until Enter is pressed
    print ("Press enter to quit,")
    print ("'q' to toggle recording,")
    print ("'p' to save data,")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)

main()