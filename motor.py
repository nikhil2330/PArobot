from gpiozero import LED,PWMLED
from time import sleep

l1 = LED("BOARD13")
l2 = LED("BOARD11")
le = PWMLED("BOARD18")
r1 = LED("BOARD15")
r2 = LED("BOARD40")
re = PWMLED("BOARD12")

def tank(l,r):
	l /= 100
	r /= 100
	if l<0:
		l1.off()
		l2.on()
		le.value = abs(l)
	elif l==0:
		l1.off()
		l2.off()
	else:
		l1.on()
		l2.off()
		le.value = abs(l)
	if r<0:
		r1.off()
		r2.on()
		re.value = abs(r)
	elif r==0:
		r1.off()
		r2.off()
	else:
		r1.on()
		r2.off()
		re.value = abs(r)
	

while True:
	for i in range(-100,100):
		tank(i,i)
		sleep(0.05)
	


