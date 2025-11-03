from gpiozero import PWMLED, LED
from time import sleep

# === Pin setup (your current wiring) ===
l_rpwm = PWMLED("BOARD33")
l_lpwm = PWMLED("BOARD31")
l_en   = LED("BOARD29")

r_rpwm = PWMLED("BOARD35")
r_lpwm = PWMLED("BOARD37")
r_en   = LED("BOARD32")

def enable_all(): l_en.on(); r_en.on()
def disable_all(): l_en.off(); r_en.off()
def stop():
    l_rpwm.value = l_lpwm.value = 0
    r_rpwm.value = r_lpwm.value = 0

enable_all()
print("Test 1: Left forward")
l_rpwm.value = 0.6; l_lpwm.value = 0
sleep(2)
stop()

print("Test 2: Left backward")
l_rpwm.value = 0; l_lpwm.value = 0.6
sleep(2)
stop()

print("Test 3: Right forward")
r_rpwm.value = 0.6; r_lpwm.value = 0
sleep(2)
stop()

print("Test 4: Right backward")
r_rpwm.value = 0; r_lpwm.value = 0.6
sleep(2)
stop()
disable_all()
