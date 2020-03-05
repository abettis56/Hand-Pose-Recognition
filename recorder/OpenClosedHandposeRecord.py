# A Hello World
# 
# This code modified from https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
# 
# Written by the Leapmotion devs as a "Hello World" for their system.

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
"""
Deprecated, used to normalize position values of joints
def normalize(coordinates, normal, colinear, cross):
    #Arrange array for ease of modification
    coordinates = np.reshape(np.asarray(coordinates), (-1, 3))
    normal = np.asarray(normal)
    colinear = np.asarray(colinear)

    #normalize translation
    coordinates = coordinates - coordinates[0]

    #reshape normal, colinear, and cross vectors for change of basis
    normal = normal.reshape((-1, 1))
    colinear = colinear.reshape((-1, 1))
    cross = cross.reshape((-1, 1))

    #normalize rotation
    for i in range(len(coordinates)):
        coordinates[i] = np.linalg.inv(np.concatenate((normal, colinear, cross), axis=1)).dot(coordinates[i])

    #convert back to list and return
    coordinates = coordinates.flatten().tolist()
    return coordinates
"""
def get_vector(vec):
    #print ("x: " + str(vec.x))
    #print ("y: " + str(vec.y))
    #print ("z: " + str(vec.z))
    
    return([vec.x, vec.y, vec.z])
"""
deprecated: Just gets 3d coords of each joint
def get_finger_joints(finger):
    #print ("----Knuckle position")
    wrist = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint)

    #print ("----Distal Metacarpal position")
    metacarpal = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint)

    #print ("----Distal Proximal Phalanx position")
    proximal = get_vector(finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint)

    #print ("----Distal Intermediate Phalanx position")
    intermediate = get_vector(finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint)

    #print ("----Distal Distal Phalanx position")
    distal = get_vector(finger.bone(Leap.Bone.TYPE_DISTAL).next_joint)
    
    finger_joints = []
    
    finger_joints.append(wrist[0])
    finger_joints.append(wrist[1])
    finger_joints.append(wrist[2])
    
    finger_joints.append(metacarpal[0])
    finger_joints.append(metacarpal[1])
    finger_joints.append(metacarpal[2])
    
    finger_joints.append(proximal[0])
    finger_joints.append(proximal[1])
    finger_joints.append(proximal[2])
    
    finger_joints.append(intermediate[0])
    finger_joints.append(intermediate[1])
    finger_joints.append(intermediate[2])
    
    finger_joints.append(distal[0])
    finger_joints.append(distal[1])
    finger_joints.append(distal[2])
    
    return finger_joints
"""
def get_finger_angles(finger, colinear, cross):
    #get vector for proximal phalanx
    vec1 = finger.bone(Leap.Bone.TYPE_PROXIMAL).direction
    proximal = np.asarray([vec1[0], vec1[1], vec1[2]])
    #get vector for intermediate phalanx
    vec2 = finger.bone(Leap.Bone.TYPE_INTERMEDIATE).direction
    intermediate = np.asarray([vec2[0], vec2[1], vec2[2]])
    #get vector for distal phalanx
    vec3 = finger.bone(Leap.Bone.TYPE_DISTAL).direction
    distal = np.asarray([vec3[0], vec3[1], vec3[2]])

    angle_set = []

    #angle between proximal phalanx and colinear vector of the hand
    proximal_adduction = vg.angle(proximal, colinear)
    angle_set.append(proximal_adduction)

    #angle between proximal phalanx and cross product between colinear and normal vectors of the hand
    proximal_flexion = vg.angle(proximal, cross)
    angle_set.append(proximal_flexion)

    #angle between intermediate and proximal phalanx
    intermediate_flexion = vg.angle(intermediate, proximal)
    angle_set.append(intermediate_flexion)

    #angle between distal and intermediate
    distal_flexion = vg.angle(distal, intermediate)
    angle_set.append(distal_flexion)

    return angle_set

def get_hand_position_data(hand):
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

    #Print hand position and get whether hand is left or right
    is_left = 0
    if hand.is_left:
        is_left = 1
        #print ("Left Hand palm position:")
    else:
        is_left = 0
        #print ("Right Hand palm position:")
    #Get palm position and normal vector and colinear vector, and calculate 
    #cross product between them for a basis vector set
    #palm_pos = get_vector(hand.palm_position)
    normal = np.asarray(get_vector(hand.palm_normal))
    colinear = np.asarray(get_vector(hand.direction))
    cross = np.cross(normal, colinear)

    #Get angles
    row = []

    row += get_finger_angles(thumb, colinear, cross)
    row += get_finger_angles(index, colinear, cross)
    row += get_finger_angles(middle, colinear, cross) 
    row += get_finger_angles(ring, colinear, cross)
    row += get_finger_angles(pinky, colinear, cross) 

    #row: [thumb_adduction, thumb_flexion, thumb_intermediate_flexion, thumb_distal_flexion,
    #      index_adduction, index_flexion, index_intermediate_flexion, index_distal_flexion,
    #      etc for all fingers...  ]

    #deprecated functionality, used to gather and normalize joint positions
    #Concat arrays to get a list of joint positions
    #row = palm_pos + thumb_pos + index_pos + middle_pos + ring_pos + pinky_pos
    #Pass joints, normal vector of palm, and colinear hand vector to normalize function to achieve rotation/translation invariance
    #row = normalize(row, normal_vector, colinear_vector, cross_vector)
    #Insert whether the hand is left at the start of the array
    #row.insert(0, is_left)

    return row

unlocked = True
data = [[], []]
dataIndex = 0
    
class SampleListener(Leap.Listener):
    def on_connect(self, controller):
        print ("Connected")

    def on_frame(self, controller):
        global unlocked
        global data
        global dataIndex
        
        if keyboard.is_pressed('space') and unlocked:
            unlocked = False
            if dataIndex == 0:
                dataIndex = 1
                print("Now recording closed hands")
            else:
                dataIndex = 0
                print("Now recording open hands")
        if keyboard.is_pressed('q') and unlocked:
            print ("Key pressed!")
            frame = controller.frame()
            if frame.is_valid:
                hands = frame.hands
                handL = None
                handR = None
                for i in range(0,2):
                    if hands[i].is_valid:
                        print ("HAND FOUND")
                        if hands[i].is_left:
                            handL = hands[i]
                            row = get_hand_position_data(handL)
                            if None not in row:
                                data[dataIndex].append(row)
                        elif hands[i].is_right:
                            handR = hands[i]
                            row = get_hand_position_data(handR)
                            if None not in row:
                                data[dataIndex].append(row)
        if keyboard.is_pressed('p') and unlocked:
            unlocked = False
            print("Starting Learning...")

            #Add rows to .csv files
            for i in range(len(data[0])):
                with open('open_raw_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[0][i])
            for i in range(len(data[1])):
                with open('closed_raw_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[1][i])

        if not keyboard.is_pressed('space') and not keyboard.is_pressed('p') and not keyboard.is_pressed('z') and not unlocked:
            unlocked = True
        

def main():
        controller = Leap.Controller()
        listener = SampleListener()
        
        # Have the sample listener receive events from the controller
        controller.add_listener(listener)
        
        unlocked = True
        
        # Keep this process running until Enter is pressed
        print ("Press enter to quit,")
        print ("'q' to record frame,")
        print ("'p' to save data and run tsne,")
        print (" or space to switch from open to closed hand poses")
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            pass
        finally:
            # Remove the sample listener when done
            controller.remove_listener(listener)

main()



