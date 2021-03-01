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


def config2comment(model_config, dataset_config):
	output = "Model configuration:\n"
	for key in model_config:
		output += "{}={}\n".format(key, model_config[key])
	output += "Dataset configuration:\n"
	for key in dataset_config:
		output += "{}={}\n".format(key, dataset_config[key])

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
					self.reset()

	def reset(self):
					self.val = 0
					self.avg = 0
					self.sum = 0
					self.count = 0

	def update(self, val, n=1):
					self.val = val
					self.sum += val * n
					self.count += n
					self.avg = self.sum / self.count
