import os

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true" 
