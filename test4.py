from gpiozero import LED
from time import sleep

en = LED("BOARD32") 
while True:
    en.on()
    print("EN ON")
    sleep(1)
    en.off()
    print("EN OFF")
    sleep(1)
