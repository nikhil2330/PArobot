from gpiozero import PWMLED
from time import sleep

a = PWMLED("BOARD33", frequency=100)
b = PWMLED("BOARD35", frequency=100)

print("Both 0.5 duty")
a.value = 0.5
b.value = 0.5
sleep(3)
print("Changing B to 0.1")
b.value = 0.1
sleep(3)
