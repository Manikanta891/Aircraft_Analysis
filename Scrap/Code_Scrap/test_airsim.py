import airsim
import time


print("Connecting to AirSim...")
# Connect
client = airsim.MultirotorClient()
client.confirmConnection()

print("Connected!")

# TAKE CONTROL
client.enableApiControl(True)
client.armDisarm(True)

time.sleep(1)

# TAKEOFF
print("Taking off...")
client.takeoffAsync().join()

time.sleep(2)

# MOVE
print("Moving...")
client.moveToPositionAsync(20, 0, -10, 5).join()

print("Done movement")