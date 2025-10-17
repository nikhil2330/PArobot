from rpi_rf import RFDevice
import time
import os
import RPi.GPIO as GPIO 

rx = RFDevice(17)
rx.enable_rx()
print("Listening for RF codes...")

last_code = None

try:
    while True:
        if rx.rx_code_timestamp != last_code:
            code = rx.rx_code
            print("Received:", code)
            if code == 11111:
                print("ON")
                # os.system("python3 start_motors.py &")  # example
            elif code == 22222:
                print("OFF")
                # os.system("pkill -f start_motors.py")   # example
            last_code = rx.rx_code_timestamp
        time.sleep(0.05)
except KeyboardInterrupt:
    rx.cleanup()
