
from gpiozero import LED, PWMLED
from time import sleep

# Left motor group
l1 = LED("BOARD13")   # IN1
l2 = LED("BOARD11")   # IN2
le = PWMLED("BOARD18") # ENA

# Right motor group
r1 = LED("BOARD22")   # IN4
r2 = LED("BOARD40")   # IN3
re = PWMLED("BOARD12") # ENB

def stop():
    l1.off(); l2.off(); le.value = 0
    r1.off(); r2.off(); re.value = 0

def forward(speed=1):
    l1.on(); l2.off(); le.value = speed
    r1.on(); r2.off(); re.value = speed

def backward(speed=1):
    l1.off(); l2.on(); le.value = speed
    r1.off(); r2.on(); re.value = speed

def left(speed=1):
    l1.off(); l2.on(); le.value = speed
    r1.on(); r2.off(); re.value = speed

def right(speed=1):
    l1.on(); l2.off(); le.value = speed
    r1.off(); r2.on(); re.value = speed

# Test sequence
print("Forward")
forward(1); sleep(2)
print("Backward")
backward(1); sleep(2)
print("Left turn")
left(); sleep(2)
print("Right turn")
right(); sleep(2)
print("Stop")
stop()

