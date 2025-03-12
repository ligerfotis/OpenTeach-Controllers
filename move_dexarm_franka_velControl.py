import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import math

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
#from allegro_hand.controller import AllegroController
from franka_arm.controller import FrankaController
from copy import deepcopy as copy



from deoxys.utils import transform_utils

from franka_arm.constants import *
from franka_arm.utils import generate_cartesian_space_min_jerk

import sys
sys.path.insert(0, "src/franka-arm-controllers")

from franka_arm.utils.timer import FrequencyTimer
from deoxys.utils.config_utils import get_default_controller_config
import argparse
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from scipy.spatial.transform import Rotation

HOME_POSITION = [0.3, 0, 0.35, 1, 0, 0, 0]
minTranslation = 0.05    #in meters
minRotationInDegree = 7.5

# Kosinussatz - zur Berechnung der Distanz der Vektorspitzen nach einer Rotation (die Vektoren haben die Länge 1)
minRotationInTranslatoryDistance = np.sqrt(2-2*np.cos(np.radians(minRotationInDegree)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")

    return parser.parse_args()

class VelocityController():
    def __init__(self):
        self.kp = 1
        self.kp_Rot = 1
        self.kd = 0
        self.kd_Rot = 0#50

        self.vDeltaTransMax = 0.001
        self.vDeltaRotMax = 0.01
        self.vMax = 0.3

        self.oldVel_X = 0
        self.oldVel_Y = 0
        self.oldVel_Z = 0
        self.oldVelRot_X = 0
        self.oldVelRot_Y = 0
        self.oldVelRot_Z = 0
        self.oldtimeStamp = 0
        self.deltaTime = 0
        self.deltaTimeMax = 0.4
        self.deltaTimeMin = 1
        self.integalError_X = 0
        self.integalError_Y = 0
        self.integalError_Z = 0
        
 
    # die Funktion wurde von ChattGPT geschrieben
    def quaternion_diff(self, quat_operatorHand, quat_robotEndeff):
        # Wandeln die Quaternions in scipy-Rotationsobjekte um
        rot_ist = R.from_quat(quat_robotEndeff)  # Quaternion-Format: [x, y, z, w]
        rot_soll = R.from_quat(quat_operatorHand)

        # Differenzrotation berechnen
        rot_diff = rot_soll * rot_ist.inv()

        # Differenz als Euler-Winkel (XYZ-Ordnung: Roll, Pitch, Yaw)
        euler_diff = rot_diff.as_euler('xyz', degrees=False)        # this is set to False, because the controller works in radiant and not in degrees

        return euler_diff

    def checkMaxVelChange(self, newVel, currentVel, vDeltaMax):
        
        if (newVel-currentVel) > vDeltaMax:
            newVel = currentVel + vDeltaMax
        if (newVel-currentVel) < -vDeltaMax:
            newVel = currentVel - vDeltaMax

        return newVel

    
    def calcMinBreakingDistance(self, currentVel, vDeltaMax):
        """
            This funciton calculates the minimal bearing distance.
        """
        estimatedStepTime = (self.deltaTimeMax-self.deltaTimeMin)/2
        minbreakingDistance = (currentVel**2 * estimatedStepTime) / (2*vDeltaMax) 
        return minbreakingDistance

    def clacControledHandVel(self, currentHandPos, currendRobotEndefPos):
        # comment on the breaking system: threoretically it should calculate when the min-distance to the goalpositioin is reached, so that it can make a perfect landing, if it makes a full break.
        # the problem with that solution is that the calculation relies on the distance which the arm moves in a certain amount of time. However, when the arm moves with a commanded velocity of
        # 0.01 into f.e. x direction the real velocity is different. As a consequence the calculated moved distance is also wrong.
        # per accident it seems to work fine with the translatory movement, which is why self.kd can be set to 0 and the arm does not oscillate, however for the rotation self.kd_Rot needs to be
        # set bigger than 0.

        currentTime = time.time()
        if self.oldtimeStamp == 0 or (currentTime-self.oldtimeStamp)>1:
            self.oldtimeStamp = currentTime
        self.deltaTime = currentTime - self.oldtimeStamp

        ## defining the position differences
        # translationi
        deltaPos_X = currentHandPos[0]-currendRobotEndefPos[0]
        deltaPos_Y = currentHandPos[1]-currendRobotEndefPos[1]
        deltaPos_Z = currentHandPos[2]-currendRobotEndefPos[2]
        # rotation
        deltaRotation = self.quaternion_diff(currentHandPos[3:], currendRobotEndefPos[3:])
        #print("deltaRotation: ", deltaRotation)
        #print("currentHandPos: ", R.from_quat(currentHandPos[3:]).as_euler('xyz', degrees=True))

        vel_X = self.kp * deltaPos_X - self.oldVel_X * self.kd
        vel_Y = self.kp * deltaPos_Y - self.oldVel_Y * self.kd
        vel_Z = self.kp * deltaPos_Z - self.oldVel_Z * self.kd

        velRot_X = self.kp_Rot * deltaRotation[0] - self.oldVelRot_X * self.kd_Rot
        velRot_Y = self.kp_Rot * deltaRotation[1] - self.oldVelRot_Y * self.kd_Rot
        velRot_Z = self.kp_Rot * deltaRotation[2] - self.oldVelRot_Z * self.kd_Rot
        
        
        ### checking the max change in velocity accelerationwise
        ## translation
        vel_X = self.checkMaxVelChange(vel_X ,self.oldVel_X, self.vDeltaTransMax)
        vel_Y = self.checkMaxVelChange(vel_Y ,self.oldVel_Y, self.vDeltaTransMax)
        vel_Z = self.checkMaxVelChange(vel_Z ,self.oldVel_Z, self.vDeltaTransMax)
        ## rotation
        velRot_X = self.checkMaxVelChange(velRot_X ,self.oldVelRot_X, self.vDeltaRotMax)
        velRot_Y = self.checkMaxVelChange(velRot_Y ,self.oldVelRot_Y, self.vDeltaRotMax)
        velRot_Z = self.checkMaxVelChange(velRot_Z ,self.oldVelRot_Z, self.vDeltaRotMax)

        # checking for max translation velocity
        if vel_X > self.vMax:
            vel_X = self.vMax
        if vel_X < -self.vMax:
            vel_X = -self.vMax
        if vel_Y > self.vMax:
            vel_Y = self.vMax
        if vel_Y < -self.vMax:
            vel_Y = -self.vMax
        if vel_Z > self.vMax:
            vel_Z = self.vMax
        if vel_Z < -self.vMax:
            vel_Z = -self.vMax

        # checking for max rotation velocity
        if velRot_X > self.vMax:
            velRot_X = self.vMax
        if velRot_X < -self.vMax:
            velRot_X = -self.vMax
        if velRot_Y > self.vMax:
            velRot_Y = self.vMax
        if velRot_Y < -self.vMax:
            velRot_Y = -self.vMax
        if velRot_Z > self.vMax:
            velRot_Z = self.vMax
        if velRot_Z < -self.vMax:
            velRot_Z = -self.vMax

        if self.deltaTime > self.deltaTimeMax:
            self.deltaTimeMax = self.deltaTime
        if self.deltaTime < self.deltaTimeMin:
            self.deltaTimeMin = self.deltaTime

        ## checking for min break distance
        # translatory
        minBreakDistance_X = self.calcMinBreakingDistance(self.oldVel_X, self.vDeltaTransMax)
        minBreakDistance_Y = self.calcMinBreakingDistance(self.oldVel_Y, self.vDeltaTransMax)
        minBreakDistance_Z = self.calcMinBreakingDistance(self.oldVel_Z, self.vDeltaTransMax)

        # rotatory
        minBreakDistanceRot_X = self.calcMinBreakingDistance(self.oldVelRot_X, self.vDeltaRotMax)
        minBreakDistanceRot_Y = self.calcMinBreakingDistance(self.oldVelRot_Y, self.vDeltaRotMax)
        minBreakDistanceRot_Z = self.calcMinBreakingDistance(self.oldVelRot_Z, self.vDeltaRotMax)

        # if the velocity in one direction is greater (or lower) than 0, if the system breaks with self.vDeltaMax, than the system should break with that value
        ## translationi
        # X
        if (deltaPos_X - minBreakDistance_X)  < 0 and self.oldVel_X  > 0:
            vel_X = self.oldVel_X - self.vDeltaTransMax
        if (deltaPos_X + minBreakDistance_X)  > 0 and self.oldVel_X  < 0:
            vel_X = self.oldVel_X + self.vDeltaTransMax
        # Y
        if (deltaPos_Y - minBreakDistance_Y)  < 0 and self.oldVel_Y  > 0:
            vel_Y = self.oldVel_Y - self.vDeltaTransMax
        if (deltaPos_Y + minBreakDistance_Y)  > 0 and self.oldVel_Y  < 0:
            vel_Y = self.oldVel_Y + self.vDeltaTransMax
        # Z
        if (deltaPos_Z - minBreakDistance_Z)  < 0 and self.oldVel_Z  > 0:
            vel_Z = self.oldVel_Z - self.vDeltaTransMax
        if (deltaPos_Z + minBreakDistance_Z)  > 0 and self.oldVel_Z  < 0:
            vel_Z = self.oldVel_Z + self.vDeltaTransMax
        ## rotation
        # X
        if (deltaRotation[0] - minBreakDistanceRot_X)  < 0 and self.oldVelRot_X  > 0:
            velRot_X = self.oldVelRot_X - self.vDeltaRotMax
        if (deltaRotation[0] + minBreakDistanceRot_X)  > 0 and self.oldVelRot_X  < 0:
            velRot_X = self.oldVelRot_X + self.vDeltaRotMax
        # Y
        if (deltaRotation[1] - minBreakDistanceRot_Y)  < 0 and self.oldVelRot_Y  > 0:
            velRot_Y = self.oldVelRot_Y - self.vDeltaRotMax
        if (deltaRotation[1] + minBreakDistanceRot_Y)  > 0 and self.oldVelRot_Y  < 0:
            velRot_Y = self.oldVelRot_Y + self.vDeltaRotMax
        # Z
        if (deltaRotation[2] - minBreakDistanceRot_Z)  < 0 and self.oldVelRot_Z  > 0:
            velRot_Z = self.oldVelRot_Z - self.vDeltaRotMax
        if (deltaRotation[2] + minBreakDistanceRot_Z)  > 0 and self.oldVelRot_Z  < 0:
            velRot_Z = self.oldVelRot_Z + self.vDeltaRotMax


        self.oldVel_X = vel_X
        self.oldVel_Y = vel_Y
        self.oldVel_Z = vel_Z

        self.oldVelRot_X = velRot_X
        self.oldVelRot_Y = velRot_Y
        self.oldVelRot_Z = velRot_Z

        return [vel_X, vel_Y, vel_Z, velRot_X, velRot_Y, velRot_Z]

class FrankaArmControl():
    def __init__(self, record=False):

        # if pub_port is set to None it will mean that
        # this will only be used for listening to franka and not commanding
        
        try:
            rospy.init_node("frankaArmController", disable_signals = True, anonymous = True)
        except:
            pass
        

        self.frankaArmCommandedHandFrame_subscriber = rospy.Subscriber('/frankaArm/commandedHandFramePosition', Float64MultiArray, self._callback_getCommandedFrankaArmPosition)
        self.frankaArmMeasuredStates_publisher = rospy.Publisher('/frankaArm/measuredRecordArmState', Float64MultiArray, queue_size=3)
        self.frankaArmCommandedStates_publisher = rospy.Publisher('/frankaArm/commandedRecordArmState', Float64MultiArray, queue_size=3)
        self.handFramePosition = None

        self.includeOffsets = False


        args = parse_args()
        self.robot_interface = FrankaInterface(
            config_root + f"/{args.interface_cfg}", use_visualizer=False
        )
        self.controller_type = "CARTESIAN_VELOCITY"
        self.controller_cfg = get_default_controller_config(controller_type=self.controller_type)

        self.myVelController = VelocityController()

        #self._init_allegro_hand_control()
        #self._init_franka_arm_control(record)

    # Controller initializers
    def _init_franka_arm_control(self, record=False):

        if record:
            print('RECORDING IN FRANKA!')

        self.franka = FrankaController(record)

    # Rostopic callback functions

    def _callback_getCommandedFrankaArmPosition(self, handFramePosition):
        self.handFramePosition = handFramePosition


    '''
    def _callback_allegro_joint_state(self, joint_state):
        self.allegro_joint_state = joint_state

    def _callback_allegro_commanded_joint_state(self, joint_state):
        self.allegro_commanded_joint_state = joint_state
    '''


    def get_arm_cartesian_coords(self):
        #current_pos, current_quat = copy(self.franka.get_cartesian_position())
        currentPosition = np.empty(7)
        current_quat, current_pos = self.robot_interface.last_eef_quat_and_pos

        currentPosition[0] = current_pos[0]
        currentPosition[1] = current_pos[1]
        currentPosition[2] = current_pos[2]
        currentPosition[3] = current_quat[0]
        currentPosition[4] = current_quat[1]
        currentPosition[5] = current_quat[2]
        currentPosition[6] = current_quat[3]

        return currentPosition

    # von ChatGPT
    def angle_between_vectors(self, a, b):
        # Skalarprodukt
        dot_product = np.dot(a, b)
        
        # Längen der Vektoren
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Winkel berechnen
        cos_theta = dot_product / (norm_a * norm_b)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clipping für numerische Stabilität
        
        return np.degrees(theta)  # Rückgabe des Winkels in Grad

    # die Funktion wurde vollständig von ChatGPT geschrieben
    def checkMinPosDiff(self, nextPos):

        newPosIsDifferent = False

        # getting the handFrame position and rotation
        frankaArmPos = self.get_arm_cartesian_coords()

        # calculating the position of the cartesian coordinate-system of the new and old handFrame - in relationship to the global coordinate system
        frankaHandFrameRot = self.quaternion_to_axes(frankaArmPos[3:])
        controllerHandFrameRot = self.quaternion_to_axes(nextPos[3:])

        # checking the difference in the translation - newPosition to oldPosition
        if np.linalg.norm(nextPos[0:3]-frankaArmPos[0:3]) > minTranslation:
            #print("translation detected")
            pass
        else:
            nextPos[0] = frankaArmPos[0]
            nextPos[1] = frankaArmPos[1]
            nextPos[2] = frankaArmPos[2]
            

        # checking the difference in the rotation - newPosition to oldPosition
        # x-axis
        xNewRotAxis = np.array([controllerHandFrameRot[0][0], controllerHandFrameRot[0][1], controllerHandFrameRot[0][2]])
        xOldRotAxis = np.array([frankaHandFrameRot[0][0], frankaHandFrameRot[0][1], frankaHandFrameRot[0][2]])
        #print("xRotAngle: ", self.angle_between_vectors(xNewRotAxis, xOldRotAxis))
        if np.linalg.norm(xNewRotAxis-xOldRotAxis) > minRotationInTranslatoryDistance:    # wenn die x Achsen um ca minRotationInTranslatoryDistance voneinander wegrotiert wurden
            newPosIsDifferent = True
            #print("x-RotDetected")

        # y-Achse
        yNewRotAxis = np.array([controllerHandFrameRot[1][0], controllerHandFrameRot[1][1], controllerHandFrameRot[1][2]])
        yOldRotAxis = np.array([frankaHandFrameRot[1][0], frankaHandFrameRot[1][1], frankaHandFrameRot[1][2]])
        #print("yRotAngle: ", self.angle_between_vectors(yNewRotAxis, yOldRotAxis))
        if np.linalg.norm(yNewRotAxis-yOldRotAxis) > minRotationInTranslatoryDistance:    # wenn die y Achsen um ca minRotationInTranslatoryDistance voneinander wegrotiert wurden
            newPosIsDifferent = True
            #print("y-RotDetected")

        # z-Achse
        zNewRotAxis = np.array([controllerHandFrameRot[2][0], controllerHandFrameRot[2][1], controllerHandFrameRot[2][2]])
        zOldRotAxis = np.array([frankaHandFrameRot[2][0], frankaHandFrameRot[2][1], frankaHandFrameRot[2][2]])
        #print("zRotAngle: ", self.angle_between_vectors(zNewRotAxis, zOldRotAxis))
        if np.linalg.norm(zNewRotAxis-zOldRotAxis) > minRotationInTranslatoryDistance:    # wenn die z Achsen um ca minRotationInTranslatoryDistance voneinander wegrotiert wurden
            newPosIsDifferent = True
            #print("z-RotDetected")

        if newPosIsDifferent == False:
            # wenn keine neue Rotationsposition detektiert wurde, dann soll die aktuelle Position zurück gegeben werden
            nextPos[3] = frankaArmPos[3]
            nextPos[4] = frankaArmPos[4]
            nextPos[5] = frankaArmPos[5]
            nextPos[6] = frankaArmPos[6]

        return nextPos

    # die Funktion wurde vollständig von ChatGPT geschrieben
    def axes_to_quaternion(self, x_axis, y_axis, z_axis):
        """
        Convert the orientation defined by the x, y, and z axes into a quaternion.
        
        Args:
            x_axis (list or np.ndarray): The x-axis of the frame.
            y_axis (list or np.ndarray): The y-axis of the frame.
            z_axis (list or np.ndarray): The z-axis of the frame.
            
        Returns:
            np.ndarray: Quaternion [x, y, z, w].
        """
        # Ensure the axes are normalized
        x_axis = np.array(x_axis) / np.linalg.norm(x_axis)
        y_axis = np.array(y_axis) / np.linalg.norm(y_axis)
        z_axis = np.array(z_axis) / np.linalg.norm(z_axis)

        # Create a 3x3 rotation matrix using the axes
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert the rotation matrix to a quaternion using scipy
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        
        return quaternion

    # die Funktion wurde vollständig von ChatGPT geschrieben
    def ensure_shortest_path(self, newPosition, measuredPosition):
        """
        Ensure the quaternion represents the shortest path relative to a reference quaternion.
        """
        if np.dot(newPosition[3:], measuredPosition[3:]) < 0:
            newPosition[3] = -newPosition[3]
            newPosition[4] = -newPosition[4]
            newPosition[5] = -newPosition[5]
            newPosition[6] = -newPosition[6]
        return newPosition

    # die Funktion wurde vollständig von ChatGPT geschrieben
    def quaternion_to_axes(self, quaternion):
        """
        Convert a quaternion into the orientation defined by the x, y, and z axes.

        Args:
            quaternion (list or np.ndarray): Quaternion [x, y, z, w].

        Returns:
            tuple: x_axis, y_axis, z_axis (each as a np.ndarray of shape (3,))
        """
        # Ensure the quaternion is normalized
        quaternion = np.array(quaternion)
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Convert the quaternion to a 3x3 rotation matrix
        rotation = R.from_quat(quaternion)  # Input should be [x, y, z, w]
        rotation_matrix = rotation.as_matrix()  # Returns a 3x3 matrix

        # Extract the axes from the rotation matrix
        x_axis = rotation_matrix[:, 0]  # First column
        y_axis = rotation_matrix[:, 1]  # Second column
        z_axis = rotation_matrix[:, 2]  # Third column

        return x_axis, y_axis, z_axis



    # Movement functions
    def move_hand(self, allegro_angles):
        self.allegro.hand_pose(allegro_angles)

    def home_hand(self):
        self.allegro.hand_pose(ALLEGRO_HOME_VALUES)

    def reset_hand(self):
        self.home_hand()

    def move_arm_joint(self, joint_angles):
        #self.franka.joint_movement(joint_angles)
        pass


# ein Codeteil, welcher eine KartesianPosition übernimmt, und sich basierend auf der 
# aktuellen Position eine neue Trajektorie ausrechnet.

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        # Moving
        start_pose = self.get_arm_cartesian_coords()
        start_time = time.time()
        '''
        poses = generate_cartesian_space_min_jerk(
            start = start_pose,
            goal = cartesian_pos,
            time_to_go = duration,
            hz = self.franka.control_freq
        )
        '''

        #print("startPosition: ", start_pose)
        #print("endPosition: ", cartesian_pos)
        #print("poses (move_arm_cartesian): ", poses)


        #for pose in poses:
        #    self.arm_control(pose)

        # Debugging the pose difference
        '''
        last_pose = self.get_arm_cartesian_coords()
        pose_error = cartesian_pos - last_pose
        debug_quat_diff = transform_utils.quat_multiply(last_pose[3:], transform_utils.quat_inverse(cartesian_pos[3:]))
        angle_diff = 180*np.linalg.norm(transform_utils.quat2axisangle(debug_quat_diff))/np.pi
        print('Absolute Pose Error: {}, Angle Difference: {}'.format(
            np.abs(pose_error[:3]), angle_diff
        ))
        '''

    def arm_control(self, cartesian_pose):
        #self.franka.cartesian_control(cartesian_pose=cartesian_pose)
        pass

    def home_arm(self):
        self.move_arm_cartesian(HOME_POSITION, duration=5)


    def reset_arm(self):
        self.home_arm()

    # Full robot commands
    def move_robot(self, allegro_angles, arm_angles):
        #self.franka.joint_movement(arm_angles, False)
        self.allegro.hand_pose(allegro_angles)

    def home_robot(self):
        #self.home_hand()
        self.home_arm() # For now we're using cartesian values

    # es muss die Position meiner Hand gemessen werden, die Zeit, welche vergangen ist zwischen den beiden Positionscommands -> dadurch wird die cmd_vel berechnet
    # ODER
    # es wird die Position meiner Hand genommen und die mit der Position des Roboterarms abgeglichen -> basierend auf der Positioinsdifferenz wird eine Geschwindigkeit berechnet, welche dem 
    # Controller übergeben wird

    def clacHandVel(self, newPosition, newTimestamp, oldPosition, oldTimeStamp):
        deltaTime = newTimestamp-oldTimeStamp
        print("deltaTime: ", deltaTime)
        print("newPosition[0]: ", newPosition[0])
        print("oldPosition[0]: ", oldPosition[0])

        Vel_X = (newPosition[0]-oldPosition[0])/deltaTime
        yVel = (newPosition[1]-oldPosition[1])/deltaTime
        zVel = (newPosition[2]-oldPosition[2])/deltaTime

        return [Vel_X, yVel, zVel, 0.0, 0.0, 0.0]


    def startCommunicationVelOnly(self):

        '''
        print("For teleoperation enter \"t\", for automatic control enter \"a\".")
        input_line = sys.stdin.read(1)
        if input_line == "t":         

            print("setting the offsets...")
            controllerHandFramePos = self.handFramePosition.data

            frankaArmPos = self.get_arm_cartesian_coords()



        elif input_line == "a":
            print("automatic control is chosen.")
        else:
            print("no valid imput - the program will stop.")
            sys.exit()
        '''

        print("starting the arm-control-loop... (press Enter to stop the loop)")
        oldHandPos = self.handFramePosition.data
        oldHandPosTimestamp = time.time()
        #print("oldHandPos: ", oldHandPos)
        
        
        try:
            while True: 

                #getting the measured position of the frankaArm
                #frankaArmPos = self.get_arm_cartesian_coords()
                
                newHandPos = self.handFramePosition.data
                #print("newHandPos: ", newHandPos)
                newHandPosTimestamp = time.time()
                handVel = self.clacHandVel(newHandPos, newHandPosTimestamp, oldHandPos, oldHandPosTimestamp)
                
                print(handVel)
               
                action = [handVel[0], handVel[1], handVel[2], 0.0, 0.0, 0.0] + [-1]
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg,
                )
                
                
                oldHandPos = newHandPos
                oldHandPosTimestamp = newHandPosTimestamp
 
                
                '''
                # only execute the control message, if at least one message is received
                if self.handFramePosition is not None:
                    # receiving the "real" hand state
                    controllerHandFramePos = self.handFramePosition.data
                    #print("controllerHandFramePos: ", controllerHandFramePos)

                    # adding the offsets to the received human controller handposition
                    # handframe translation/position
                    if self.includeOffsets == True:
                        newHandPos = np.array([
                            controllerHandFramePos[0] + xTransOffset, 
                            controllerHandFramePos[1] + yTransOffset, 
                            controllerHandFramePos[2] + zTransOffset,
                            controllerHandFramePos[3] + q1RotOffset,
                            controllerHandFramePos[4] + q2RotOffset,
                            controllerHandFramePos[5] + q3RotOffset,
                            controllerHandFramePos[6] + q4RotOffset
                        ])
                    else:
                        newHandPos = np.array(controllerHandFramePos)

                    print("newHandPos: ", newHandPos)
                    newHandPos = self.checkMinPosDiff(newHandPos)
                    newHandPos = self.ensure_shortest_path(newHandPos, frankaArmPos)
                    #print("newHandPos: ", newHandPos)

                    self.move_arm_cartesian([newHandPos[0], newHandPos[1], newHandPos[2], newHandPos[3], newHandPos[4], newHandPos[5], newHandPos[6]], duration=0.1)
                    #self.move_arm_cartesian([newHandPos[0], newHandPos[1], newHandPos[2], 1, 0, 0, 0], duration=1)
                else: 
                    print("no ros-arm-command message received ...")
                    time.sleep(0.1)
            
                # sending the ArmPositionData to the OT-Framework ...
                msg = Float64MultiArray()
                msg.data = frankaArmPos
                #self.frankaArmMeasuredStates_publisher.publish(msg)
                if self.handFramePosition is not None:
                    msg = Float64MultiArray()
                    msg.data = newHandPos
                    #self.frankaArmCommandedStates_publisher.publish(msg)
                '''
        except KeyboardInterrupt:
            self.trajektorieExecThreadClass.stop()
            print("Die Schleife und das Programm wurden beendet.")






    def startCommunication(self):

        print("For teleoperation enter \"t\", for automatic control enter \"a\".")
        input_line = sys.stdin.read(1)
        if input_line == "t":         

            while self.handFramePosition is None:
                print("no message received - self.handFramePosition == None")
                time.sleep(0.1)

            print("setting the offsets...")
            controllerHandFramePos = self.handFramePosition.data
            frankaArmPos = self.get_arm_cartesian_coords()

            print("setting the translatory offsets ...")
            xTransOffset = frankaArmPos[0] - controllerHandFramePos[0]
            yTransOffset = frankaArmPos[1] - controllerHandFramePos[1]
            zTransOffset = frankaArmPos[2] - controllerHandFramePos[2]

            print("setting the rotatory offsets ...")
            q1RotOffset = frankaArmPos[3] - controllerHandFramePos[3]
            q2RotOffset = frankaArmPos[4] - controllerHandFramePos[4]
            q3RotOffset = frankaArmPos[5] - controllerHandFramePos[5]
            q4RotOffset = frankaArmPos[6] - controllerHandFramePos[6]
            self.includeOffsets = True

        elif input_line == "a":
            print("automatic control is chosen.")
        else:
            print("no valid imput - the program will stop.")
            sys.exit()

        print("starting the arm-control-loop... (press Enter to stop the loop)")
        try:
            while True: 

                #getting the measured position of the frankaArm
                frankaArmPos = self.get_arm_cartesian_coords()
                newHandPos = frankaArmPos

                # only execute the control message, if at least one message is received
                if self.handFramePosition is not None:
                    # receiving the "real" hand state
                    controllerHandFramePos = self.handFramePosition.data

                    if self.includeOffsets:
                        newHandPos = np.array([
                            controllerHandFramePos[0] + xTransOffset, 
                            controllerHandFramePos[1] + yTransOffset, 
                            controllerHandFramePos[2] + zTransOffset,
                            controllerHandFramePos[3] + q1RotOffset,
                            controllerHandFramePos[4] + q2RotOffset,
                            controllerHandFramePos[5] + q3RotOffset,
                            controllerHandFramePos[6] + q4RotOffset
                        ])
                    else: 
                        newHandPos = np.array(controllerHandFramePos)
                    #print("controllerHandFramePos: ", controllerHandFramePos)

                    handVel = self.myVelController.clacControledHandVel(newHandPos, frankaArmPos)

                    action = [handVel[0], handVel[1], handVel[2], handVel[3], handVel[4], handVel[5]] + [-1]
                    self.robot_interface.control(
                        controller_type=self.controller_type,
                        action=action,
                        controller_cfg=self.controller_cfg,
                    )
                   
                else: 
                    print("no ros-arm-command message received ...")
                    time.sleep(0.1)
            
                # sending the ArmPositionData to the OT-Framework ...
                msg = Float64MultiArray()
                msg.data = frankaArmPos
                self.frankaArmMeasuredStates_publisher.publish(msg)
                if self.handFramePosition is not None:
                    msg = Float64MultiArray()
                    msg.data = newHandPos
                    self.frankaArmCommandedStates_publisher.publish(msg)
                
        except KeyboardInterrupt:
            self.trajektorieExecThreadClass.stop()
            print("Die Schleife und das Programm wurden beendet.")

            #print("handRotationController: ", handRotationController)


    def moveToHomePosition(self):
        
        reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
        ''' # das ist ein möglicher Start fürs Buchöffnen
        reset_joint_positions = [
            0.09162008114028396,
            0.55,
            -0.01990020486871322,
            -1.7,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
        '''
        reset_joints_to(self.robot_interface, reset_joint_positions)

    def makeSmallInitialMovement(self):

        for i in range(10):
            action = [0.001, 0, 0, 0, 0, 0] + [-1]

            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )

        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1]
        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )


if __name__ == '__main__':
    frankaArm = FrankaArmControl()
        

    frankaArm.makeSmallInitialMovement()

    print("moving the robot to the home position (takes a view 5 seconds) ...")
    frankaArm.moveToHomePosition()     

    print("starting the communication between the OT-Framework and the FrankaArm ...")
    frankaArm.startCommunication()

    