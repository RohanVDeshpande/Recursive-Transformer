import sys

def confirm(msg):
	print(msg)
	while True:
	    reply = str(input("Are you sure you want to proceed? [y/n]")).lower().strip()
	    if reply == "y":
	        print("Proceeding...")
	        break
	    elif reply == "n":
	        print("Will not proceed...")
	        sys.exit(1)