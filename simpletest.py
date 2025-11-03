from gpiozero import LED
from time import sleep

led = LED(17)

print("Blink test running...")
for i in range(5):
    led.toggle()
    print("Toggle:", i + 1)
    sleep(1)
print("Done.")
