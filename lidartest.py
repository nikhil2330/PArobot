import ydlidar
import math
import matplotlib.pyplot as plt

ydlidar.os_init()
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 6.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

if not laser.initialize():
    print("Failed to initialize LiDAR")
    exit(1)

if not laser.turnOn():
    print("Failed to start scanning")
    exit(1)

scan = ydlidar.LaserScan()

plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(True)
lidar_point, = ax.plot(0, 0, 'ro')

try:
	while True:
		if laser.doProcessSimple(scan):
			front_points = [p for p in scan.points if -10 <= p.angle <= 10 and p.range > 0.15]
			if front_points:
				xs = [p.range * math.cos(p.angle) for p in front_points]
				ys = [p.range * math.sin(p.angle) for p in front_points]
				ax.clear()
				ax.set_aspect('equal')
				ax.set_xlim(-5, 5)
				ax.set_ylim(-5, 5)
				ax.grid(True)
				ax.plot(xs, ys, 'b.', markersize=2) 
				ax.plot(0, 0, 'ro', markersize=8)    
				plt.pause(0.01)
			else:
				print("No Front Points")
		else:
			print("⚠️ No scan data")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    laser.turnOff()
    laser.disconnecting()
    plt.ioff()
    plt.show() 
