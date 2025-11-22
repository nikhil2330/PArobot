from gpiozero import PWMLED, LED
from time import sleep

# Left BTS7960
l_rpwm = PWMLED("BOARD33", frequency=200)   # GPIO13
l_lpwm = PWMLED("BOARD31", frequency=200)   # GPIO6
l_en   = LED("BOARD29")      # GPIO5

# Right BTS7960
r_rpwm = PWMLED("BOARD35", frequency=200)   # GPIO19
r_lpwm = PWMLED("BOARD37", frequency=200)   # GPIO26
r_en   = LED("BOARD32")      # GPIO12

def enable_all():
    l_en.on(); r_en.on()

def disable_all():
    l_en.off(); r_en.off()

def stop():
    l_rpwm.value = l_lpwm.value = 0
    r_rpwm.value = r_lpwm.value = 0

def forward(speed=0.6):
    l_rpwm.value = speed; l_lpwm.value = 0
    r_rpwm.value = speed; r_lpwm.value = 0

def backward(speed=0.6):
    l_rpwm.value = 0; l_lpwm.value = speed
    r_rpwm.value = 0; r_lpwm.value = speed

def left(speed=0.6):
    l_rpwm.value = 0; l_lpwm.value = speed
    r_rpwm.value = speed; r_lpwm.value = 0

def right(speed=0.6):
    l_rpwm.value = speed; l_lpwm.value = 0
    r_rpwm.value = 0; r_lpwm.value = speed

# ==== TEST SEQUENCE ====
enable_all()
print("Forward");  forward(0.5);  sleep(2)
print("Backward"); backward(0.5); sleep(2)
print("Left");     left(0.5);     sleep(2)
print("Right");    right(0.5);    sleep(2)
print("Stop");     stop()
disable_all()
