import subprocess

file = "testing.txt"

# Define your commands
commands = [
    "ssh login1 sbatch /home1/08809/tg881088/NanoDotOptimization/" + file,
]

# Execute each command
for cmd in commands:
    result = subprocess.run(cmd, shell=True, check=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        break