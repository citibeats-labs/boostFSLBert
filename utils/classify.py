import numpy as np

def get_predictions(probabilities, p=0.5):
	"""
	Objective: from probabilities or similarities get the predictions depending on the threshold we put
	
	Inputs:
		- probabilities, np.array: probabilities
		- p, float: probability threshold
	Outputs:
		- preds np.array: same shape of proababilities but with predictions only
	"""
	
	preds = np.sign((probabilities > p) * probabilities)

	return preds