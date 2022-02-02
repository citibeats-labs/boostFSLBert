def evaluate(y_true, y_preds):
	"""
	Objective: get the evaluations metrics from the predictions
	Inputs:
		- y_true, np.array: true intents
		- y_preds, np.array: predictions of the model
	Outputs:
		- metrics, dict: a dictionary with the precision, recall accuracy and f1
	"""
	metrics = {'tn': 0, 'fn': 0, 'tp': 0, 'fp': 0}
	
	metrics['accuracy'] = round((y_true == y_preds).mean() * 100, 2)
	
	for i, pred in enumerate(y_preds):
		true = y_true[i]
		if (true, pred) == (0, 0):
			metrics['tn'] += 1
		elif (true, pred) == (1, 1):
			metrics['tp'] += 1
		elif (true, pred) == (0, 1):
			metrics['fp'] += 1
		else:
			metrics['fn'] += 1
	
	num = metrics['tp']
	denum_r = metrics['tp'] + metrics['fn']
	denum_p = metrics['tp'] + metrics['fp']
			
	r = round(num / denum_r * 100, 2) if denum_r > 0 else 0
	p = round(num / denum_p * 100, 2) if denum_p > 0 else 0
	
	metrics['precision'] = p
	metrics['recall'] = r
	metrics['f1'] = 2 * r * p / (r + p) if (r + p) > 0 else 0
	
	return metrics