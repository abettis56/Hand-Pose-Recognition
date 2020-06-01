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

#Used to normalize position values of joints
def normalize(coordinates, normal, colinear, cross):
    #Arrange array for ease of modification
    coordinates = np.asarray(coordinates)
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

def PCA_normalize(coordinates):
    coordinates = np.asarray(coordinates)
    coordinates = coordinates - coordinates[0]
    pca_data = PCA(n_components=3).fit_transform(coordinates)
    pca_data = pca_data.flatten().tolist()
    return pca_data

def experimental_helper(finger):

    metacarpal = finger.bone(Leap.Bone.TYPE_METACARPAL)

    proximal = finger.bone(Leap.Bone.TYPE_PROXIMAL)

    intermediate = finger.bone(Leap.Bone.TYPE_INTERMEDIATE)

    distal = finger.bone(Leap.Bone.TYPE_DISTAL)

    joints = []
    joints = joints + experimental_positions(metacarpal, proximal)
    joints = joints + experimental_positions(proximal, intermediate)
    joints = joints + experimental_positions(intermediate, distal)

    return joints

#This function accepts two Leap bone objects and gets the position of the most distal involved joint
def experimental_positions(prox_bone, dist_bone):
    #Get all the relevant data from the more proximal of the two bones
    bone_pos = get_vector(prox_bone.center)
    x_basis = get_vector(prox_bone.basis.x_basis)
    y_basis = get_vector(prox_bone.basis.y_basis)
    z_basis = get_vector(prox_bone.basis.z_basis)
    prox_joint = get_vector(prox_bone.next_joint)

    #Now, get distal joint (or fingertip if it's the distal phalanx) from more distal bone
    dist_joint = get_vector(dist_bone.next_joint)

    #make numpy arrays out of ALL the things
    bone_pos = np.asarray(bone_pos)
    x_basis = np.asarray(x_basis).reshape((-1, 1)) * -1
    y_basis = np.asarray(y_basis).reshape((-1, 1)) * -1
    z_basis = np.asarray(z_basis).reshape((-1, 1)) * -1
    prox_joint = np.asarray(prox_joint)
    dist_joint = np.asarray(dist_joint)

    #Normalize position, center on bone_pos
    dist_joint = dist_joint - bone_pos
    prox_joint = prox_joint - bone_pos

    #rotate to change basis to xyz_basis
    dist_joint = np.linalg.inv(np.concatenate((x_basis, y_basis, z_basis), axis=1)).dot(dist_joint)
    prox_joint = np.linalg.inv(np.concatenate((x_basis, y_basis, z_basis), axis=1)).dot(prox_joint)

    #Normalize position again, switching to proximal joint as the centerpoint
    dist_joint = (dist_joint - prox_joint)

    #dist_joint is now a position vector from the origin, normalized.
    """For when this was used to get angles
    #Use VG's angle calculations to get the angle in radians.
    #Specify "look" = unit basis vectors x = <1, 0, 0> y = <0, 1, 0> z= <0, 0, 1>
    #to get angles only in an individual plane
    #angles = []
    #yz-plane (x-rot)
    #angles.append(vg.signed_angle(np.array([0, 0, -1]), dist_joint, look=np.array([1, 0, 0]), units='deg'))
    #xz-plane (y-rot)
    #angles.append(vg.signed_angle(np.array([0, 0, -1]), dist_joint, look=np.array([0, 1, 0]), units='deg'))
    #xy-plane (z-rot)
    #angles.append(vg.signed_angle(np.array([0, -1, 0]), dist_joint, look=np.array([0, 0, 1]), units='deg'))
    """
    return dist_joint.tolist()

def get_vector(vec):
    #print ("x: " + str(vec.x))
    #print ("y: " + str(vec.y))
    #print ("z: " + str(vec.z))
    
    return([vec.x, vec.y, vec.z])

#Gets 3d coords of each joint
def get_finger_joints(finger):
    #print ("----Knuckle position")
    #wrist = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint)

    #print ("----MCP position")
    metacarpal = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint)

    #print ("----Distal Proximal Phalanx position")
    proximal = get_vector(finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint)

    #print ("----Distal Intermediate Phalanx position")
    intermediate = get_vector(finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint)

    #print ("----Distal Distal Phalanx position")
    distal = get_vector(finger.bone(Leap.Bone.TYPE_DISTAL).next_joint)
    
    finger_joints = []
    """
    finger_joints.append(wrist[0])
    finger_joints.append(wrist[1])
    finger_joints.append(wrist[2])
    """
    finger_joints.append([])
    finger_joints[0].append(metacarpal[0])
    finger_joints[0].append(metacarpal[1])
    finger_joints[0].append(metacarpal[2])
    
    finger_joints.append([])
    finger_joints[1].append(proximal[0])
    finger_joints[1].append(proximal[1])
    finger_joints[1].append(proximal[2])
    
    finger_joints.append([])
    finger_joints[2].append(intermediate[0])
    finger_joints[2].append(intermediate[1])
    finger_joints[2].append(intermediate[2])
    
    finger_joints.append([])
    finger_joints[3].append(distal[0])
    finger_joints[3].append(distal[1])
    finger_joints[3].append(distal[2])
    
    return finger_joints

def get_finger_angles(finger, colinear, normal):
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
    proximal_flexion = vg.angle(proximal, normal)
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
    palm_pos = [get_vector(hand.palm_position)]
    normal = np.asarray(get_vector(hand.palm_normal))
    colinear = np.asarray(get_vector(hand.direction))
    cross = np.cross(normal, colinear)
    #negate the vectors
    normal = np.negative(normal)
    colinear = np.negative(colinear)
    cross = np.negative(cross)

    #Get angles
    row = []

    #row += get_finger_angles(thumb, colinear, normal)
    #row += get_finger_angles(index, colinear, normal)
    #row += get_finger_angles(middle, colinear, normal) 
    #row += get_finger_angles(ring, colinear, normal)
    #row += get_finger_angles(pinky, colinear, normal)

    #thumb_pos = get_finger_joints(thumb)
    #index_pos = get_finger_joints(index)
    #middle_pos = get_finger_joints(middle)
    #ring_pos = get_finger_joints(ring)
    #pinky_pos = get_finger_joints(pinky)

    row += experimental_helper(thumb)
    row += experimental_helper(index)
    row += experimental_helper(middle)
    row += experimental_helper(ring)
    row += experimental_helper(pinky)

    #Check if row has erroneous results (+/- 50 currently allowed)
    #Rows containing None are not accepted
    thisArray = np.asarray(row)
    if(np.amax(thisArray) >= 50 or np.amin(thisArray) <= -50):
        row.append(None)

    #row: [thumb_adduction, thumb_flexion, thumb_intermediate_flexion, thumb_distal_flexion,
    #      index_adduction, index_flexion, index_intermediate_flexion, index_distal_flexion,
    #      etc for all fingers...  ]

    #used to gather and normalize joint positions with PCA
    #Concat arrays to get a list of joint positions
    #row = palm_pos + thumb_pos + index_pos + middle_pos + ring_pos + pinky_pos
    #Pass joints, normal vector of palm, and colinear hand vector to normalize function to achieve rotation/translation invariance
    #row = normalize(row, normal, colinear, cross)
    #row = PCA_normalize(row)
    #Insert whether the hand is left at the start of the array
    #row.insert(0, is_left)

    return row

def printHand(row):
    print("Thumb first joint: " + str(row[0]) + " ," + str(row[1]) + " ," + str(row[2]) + '\n' +
    "Thumb second joint: " + str(row[3]) + " ," + str(row[4]) + " ," + str(row[5]) + '\n' +
    "Thumb end: " + str(row[6]) + " ," + str(row[7]) + " ," + str(row[8]) + '\n' +

    "Index first joint: " + str(row[9]) + " ," + str(row[10]) + " ," + str(row[11]) + '\n' +
    "Index second joint: " + str(row[12]) + " ," + str(row[13]) + " ," + str(row[14]) + '\n' +
    "Index end: " + str(row[15]) + " ," + str(row[16]) + " ," + str(row[17]) + '\n' +

    "Middle first joint: " + str(row[18]) + " ," + str(row[19]) + " ," + str(row[20]) + '\n' +
    "Middle second joint: " + str(row[21]) + " ," + str(row[22]) + " ," + str(row[23]) + '\n' +
    "Middle end: " + str(row[24]) + " ," + str(row[25]) + " ," + str(row[26]) + '\n' +

    "Ring first joint: " + str(row[27]) + " ," + str(row[28]) + " ," + str(row[29]) + '\n' +
    "Ring second joint: " + str(row[30]) + " ," + str(row[31]) + " ," + str(row[32]) + '\n' +
    "Ring end: " + str(row[33]) + " ," + str(row[34]) + " ," + str(row[35]) + '\n' +

    "Pinky first joint: " + str(row[36]) + " ," + str(row[37]) + " ," + str(row[38]) + '\n' +
    "Pinky second joint: " + str(row[39]) + " ," + str(row[40]) + " ," + str(row[41]) + '\n' +
    "Pinky end: " + str(row[42]) + " ," + str(row[43]) + " ," + str(row[44]) + '\n')

unlocked = True
data = [[], [], [], [], [], [], [], [], [], []]
dataIndex = 0
    
class SampleListener(Leap.Listener):
    def on_connect(self, controller):
        print ("Connected")

    def on_frame(self, controller):
        global unlocked
        global data
        global dataIndex
        
        #print current data readout
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
                        printHand(row)
                    elif hands[i].is_right:
                        handR = hands[i]
                        row = get_hand_position_data(handR)
                        printHand(row)


        #check inputs and react accordingly
        if keyboard.is_pressed('0') and unlocked:
            unlocked = False
            dataIndex = 0
            print("Now recording index flexing")
        if keyboard.is_pressed('1') and unlocked:
            unlocked = False
            dataIndex = 1
            print("Now recording middle flexing")
        if keyboard.is_pressed('2') and unlocked:
            unlocked = False
            dataIndex = 2
            print("Now recording ring flexing")
        if keyboard.is_pressed('3') and unlocked:
            unlocked = False
            dataIndex = 3
            print("Now recording pinky flexing")
        if keyboard.is_pressed('4') and unlocked:
            unlocked = False
            dataIndex = 4
            print("Now recording thumb flexing")
        if keyboard.is_pressed('5') and unlocked:
            unlocked = False
            dataIndex = 5
            print("Now recording index flexed")
        if keyboard.is_pressed('6') and unlocked:
            unlocked = False
            dataIndex = 6
            print("Now recording middle flexed")
        if keyboard.is_pressed('7') and unlocked:
            unlocked = False
            dataIndex = 7
            print("Now recording ring flexed")
        if keyboard.is_pressed('8') and unlocked:
            unlocked = False
            dataIndex = 8
            print("Now recording pinky flexed")
        if keyboard.is_pressed('9') and unlocked:
            unlocked = False
            dataIndex = 9
            print("Now recording ALL")

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
            print("Saving...")

            #Add rows to .csv files
            for i in range(len(data[0])):
                with open('index_flex_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[0][i])
            for i in range(len(data[1])):
                with open('middle_flex_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[1][i])
            for i in range(len(data[2])):
                with open('ring_flex_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[2][i])
            for i in range(len(data[3])):
                with open('pinky_flex_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[3][i])
            for i in range(len(data[4])):
                with open('thumb_flex_out.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[4][i])
            for i in range(len(data[5])):
                with open('index_flexed.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[5][i])
            for i in range(len(data[6])):
                with open('middle_flexed.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[6][i])
            for i in range(len(data[7])):
                with open('ring_flexed.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[7][i])
            for i in range(len(data[8])):
                with open('pinky_flexed.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[8][i])
            for i in range(len(data[9])):
                with open('hand_manifold.csv', 'ab') as file:
                    writer = csv.writer(file)
                    writer.writerow(data[9][i])

            print("Saved.")

        if (not keyboard.is_pressed('0') and
        not keyboard.is_pressed('1') and
        not keyboard.is_pressed('2') and
        not keyboard.is_pressed('3') and
        not keyboard.is_pressed('4') and
        not keyboard.is_pressed('5') and
        not keyboard.is_pressed('6') and
        not keyboard.is_pressed('7') and
        not keyboard.is_pressed('8') and
        not keyboard.is_pressed('9') and
        not keyboard.is_pressed('p') and
        not unlocked):
            unlocked = True

        #clear screen
        sleep(.017)
        clear()
        

def main():
        controller = Leap.Controller()
        listener = SampleListener()
        
        # Have the sample listener receive events from the controller
        controller.add_listener(listener)
        
        unlocked = True
        
        # Keep this process running until Enter is pressed
        print ("Press enter to quit,")
        print ("'q' to record frame,")
        print ("'p' to save data,")
        print (" or 0-4 to switch between finger flexion")
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            pass
        finally:
            # Remove the sample listener when done
            controller.remove_listener(listener)

main()



