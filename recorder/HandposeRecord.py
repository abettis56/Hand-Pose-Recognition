# A Hello World
# 
# This code acquired directly from https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
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

def get_vector(vec):
    print "x: " + str(vec.x)
    print "y: " + str(vec.y)
    print "z: " + str(vec.z)
    
    return([vec.x, vec.y, vec.z])

def get_finger_joints(finger):
    print "----Knuckle position"
    knuckle = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint)

    print "----Distal Metacarpal position"
    metacarpal = get_vector(finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint)

    print "----Distal Proximal Phalanx position"
    proximal = get_vector(finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint)

    print "----Distal Intermediate Phalanx position"
    intermediate = get_vector(finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint)

    print "----Distal Distal Phalanx position"
    distal = get_vector(finger.bone(Leap.Bone.TYPE_DISTAL).next_joint)
    
    finger_joints = []
    
    finger_joints.append(knuckle[0])
    finger_joints.append(knuckle[1])
    finger_joints.append(knuckle[2])
    
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

def get_hand_position_data(hand):
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

    #Print hand position
    is_left = 0
    if hand.is_left:
        is_left = 1
        print "Left Hand palm position:"
    else:
        is_left = 0
        print "Right Hand palm position:"
    palm_pos = get_vector(hand.palm_position)

    print "\nThumb joint positions:"
    thumb_pos = get_finger_joints(thumb)
    print "\nIndex joint positions:"
    index_pos = get_finger_joints(index)
    print "\nMiddle joint positions:"
    middle_pos = get_finger_joints(middle)
    print "\nRing joint positions:"
    ring_pos = get_finger_joints(ring)
    print "\nPinky joint positions:"
    pinky_pos = get_finger_joints(pinky)
    
    row = palm_pos + thumb_pos + index_pos + middle_pos + ring_pos + pinky_pos
    row.insert(0, is_left)
    
    print(row)
    with open('out.csv', 'ab') as file:
        writer = csv.writer(file)
        writer.writerow(row)

unlocked = True
    
class SampleListener(Leap.Listener):

    def on_connect(self, controller):
        print "Connected"

    def on_frame(self, controller):
        global unlocked
        #try:
        if keyboard.is_pressed('q') and unlocked:
            unlocked = False
            print "Key pressed!"
            frame = controller.frame()
            if frame.is_valid:
                print "frame collected!"
                hands = frame.hands
                handL = None
                handR = None
                for i in range(0,2):
                    if hands[i].is_valid:
                        print "HAND FOUND"
                        if hands[i].is_left:
                            handL = hands[i]
                            row = get_hand_position_data(handL)
                        elif hands[i].is_right:
                            handR = hands[i]
                            row = get_hand_position_data(handR)
        elif not keyboard.is_pressed('q') and not unlocked:
            unlocked = True
        

def main():
        controller = Leap.Controller()
        listener = SampleListener()
        
        # Have the sample listener receive events from the controller
        controller.add_listener(listener)
        
        unlocked = True
        
        # Keep this process running until Enter is pressed
        print "Press enter to quit or 'q' to record frame"
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            pass
        finally:
            # Remove the sample listener when done
            controller.remove_listener(listener)

main()



