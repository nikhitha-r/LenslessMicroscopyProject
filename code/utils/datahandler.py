import csv
import numpy as np

def loadResultFiles(fpath):
	with open(fpath) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=' ')
		cols=next(csv_reader)
		ncol=len(cols)-1
		line_count = 0
		data={}
		print('name of columns:')
		for r in range(0,ncol):
			#print(cols[r+1])
			nname=cols[r+1]
			data[nname]=[]
		
		for row in csv_reader:
			for r in range(0,ncol):
				nname=cols[r+1]
				if len(row):
					if(isinstance(row[r], (int, float))):
						data[nname].append(float(row[r]))
					else:
						data[nname].append(row[r])
					line_count += 1
			
		print('Processed lines:',line_count)
		return data


def getDataAsNumpy(dats):
	ti_temp = np.array(dats['TIME[SEC]'])
	conf_temp = np.array(dats['AREA[%]'])
	ct_temp = np.array(dats['CONTOUR[PX]'])
	data = {'time' : ti_temp.astype(np.float),
		'confluency' : conf_temp.astype(np.float),
		'contour' : ct_temp.astype(np.float)}
	return data

