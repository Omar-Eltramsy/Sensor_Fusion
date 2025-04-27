import serial.tools.list_ports
import time

serial_port = 'COM5'
baud_rate = 9600
serial_object = serial.Serial(serial_port, baud_rate)
data = []

for _ in range(38):
    if serial_object.in_waiting:  # Check if data is available
        packet = serial_object.readline()
        val = packet.decode('utf-8').strip('\n').strip('\r') 
        data.append(val) 
        print(val)
    else:
        print("No data available at the moment")
    time.sleep(1)  

# Close the serial connection after use
serial_object.close()

print("Data has been successfully Readen")