import time
from pathlib import Path

import numpy as np
import viser
import serial
import serial.tools.list_ports
import viser.transforms as tf
import trimesh
from collections import deque

import struct

step_through = False
breadboardScale = 2

ports = serial.tools.list_ports.comports()
for port in ports:
    if port.device == "/dev/ttyACM0" or port.device == "/dev/ttyACM1" or port.device == "/dev/ttyACM2":
        SerialPort = serial.Serial(port=port.device, baudrate = 115200)
SerialPort.flushInput()
SerialPort.flushOutput()
time.sleep(1)

def step(inputQueue, breadboard, pointCloud, newCloud):
    cloudUpdated = False
    while cloudUpdated == False:
        identifier = inputQueue.popleft()
        if identifier == 0:
            pointCloudPoints = inputQueue.popleft()
            print("Main point cloud updated-------------------------------------------------")
            pointCloud.points = np.append(pointCloud.points, pointCloudPoints)
            newCloud.points = pointCloudPoints #Uses newCloud and recolors to indicate new addition to the total pointCloud
            cloudUpdated = True
        elif identifier == 1:
            newCloudPoints = inputQueue.popleft()
            print("Iteration begun----------------------------------------------------------")
            newCloud.points = newCloudPoints
            cloudUpdated = True
        elif identifier == 2:
            poseData = inputQueue.popleft()
            print("Quaternion = ", poseData[0])
            print("Position = ", poseData[1])
            print("Velocity = ", poseData[2])
            breadboard.wxyz = poseData[0]
            breadboard.position = poseData[1]
        else:
            inputQueue.popleft()
            print("identifier failure")
    
def main():
    START_MARKER = 0x7E
    END_MARKER = 0x7F
    ESCAPE_BYTE = 0x7D
    in_packet = False
    descriptorString = ""
    floatBytes = []
    server = viser.ViserServer()
    mesh = trimesh.load_mesh(str(Path(__file__).parent / "breadboard.obj"))
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_scale(0.001 * breadboardScale)
    inputQueue = deque()
    breadboard = server.scene.add_mesh_simple(
        "/Breadboard",
        vertices = mesh.vertices,
        faces = mesh.faces,
        wxyz = tf.SO3.from_quaternion_xyzw(xyzw = np.array([0.0, 0.0, 0.0, 1.0])).wxyz,
        position = (0, 0, 0)
    )
    pointCloud = server.scene.add_point_cloud(
        "/Points",
        points = np.ndarray((0, 3)),
        colors = (82, 46, 242),
        point_size = 0.01
    )
    newCloud = server.scene.add_point_cloud(
        "/NewCloud",
        points = np.ndarray((0,3)),
        colors = (255, 96, 19),
        point_size = 0.015
    )
    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")
    if (step_through):
        button = server.gui.add_button(label="Step", color="indigo")
        button.on_click(lambda _:(step(inputQueue, breadboard, pointCloud, newCloud)))
    while True:
        try:
            Input = SerialPort.read(1)[0]
        except serial.serialutil.SerialException:
            while True:
                time.sleep(1) # Pause indefinately if serial port is unplugged
        if not in_packet:
            if Input == START_MARKER :
                in_packet = True
            elif Input == 0x0A: #End of line, this implies serial_write was called with only a string
                # convert floatBytes list to float list
                if (len(floatBytes) % 4 == 0 and len(floatBytes) != 0) :
                    num_floats = len(floatBytes) // 4
                    floats = struct.unpack(f'{num_floats}f', bytes(floatBytes))
                    if descriptorString.strip() == "newPts:":
                        pointCloudPoints = np.ndarray((0,3))
                        temp_list = pointCloud.points.tolist()
                        if step_through:
                            temp_list = pointCloudPoints.tolist()
                        for i in range(0, (int(len(floats)/3)-1)): # Start at 1 because 0th index is "NewPts"
                            try:
                                if step_through:
                                    pointCloudPoints = np.append(pointCloudPoints, [floats[3*i], floats[3*i+1], floats[3*i+2]])
                                else:
                                    pointCloud.points = np.append(pointCloud.points, [floats[3*i], floats[3*i+1], floats[3*i+2]])
                            except ValueError, IndexError:
                                SerialPort.flushInput()
                                SerialPort.flushOutput()
                        if step_through:
                            inputQueue.append(0)
                            inputQueue.append(pointCloudPoints)
#                    elif descriptorString.strip() == "oldPts:" and step_through:
#                        newCloudPoints = np.ndarray((0,3))
#                        temp_list = newCloudPoints.tolist()
#                        for i in range(0, (int(len(floats)/3)-1)): # Start at 1 because 0th index is "NewPts"
#                            try:
#                                temp_list.append([floats[3*i], floats[3*i+1], floats[3*i+2]])
#                            except ValueError, IndexError:
#                                SerialPort.flushInput()
#                                SerialPort.flushOutput()
#                        newCloudPoints = np.append(newCloudPoints, temp_list)
#                        if step_through:
#                            inputQueue.append(1)
#                            inputQueue.append(newCloudPoints)
                    elif descriptorString.strip() == "inertialUpdate:":
                        try:
                            if step_through:
                                breadboardWxyz = tf.SO3.from_quaternion_xyzw(xyzw = np.array([floats[1], floats[2], floats[3], floats[0]])).wxyz
                                breadboardPosition = (floats[4], floats[5], floats[6])
                                breadboardVelocity = (floats[7], floats[8], floats[9])
                                inputQueue.append(2)
                                inputQueue.append([breadboardWxyz, breadboardPosition, breadboardVelocity])
                            else:
                                breadboard.wxyz = tf.SO3.from_quaternion_xyzw(xyzw = np.array([floats[1], floats[2], floats[3], floats[0]])).wxyz
                                breadboard.position = (floats[4], floats[5], floats[6])
                        except ValueError, IndexError:
                            SerialPort.flushInput()
                            SerialPort.flushOutput()
                descriptorString = ""
                floatBytes = []
            else:
                descriptorString += chr(Input)
        else:
            if Input == END_MARKER:
                in_packet = False
            else:
                if Input == ESCAPE_BYTE:
                    Input = SerialPort.read(1)[0] ^ 0x20
                floatBytes.append(Input)

if __name__=="__main__":
    main()