"""Helper functions used in Data Analysis notebook."""
import numpy as np
import matplotlib.pyplot as plt

def dataScatter(tX, y, saveDir=None):
	N = np.shape(tX)[1]
	for i in range(N):
		print("Feature " + str(i))
		f = tX[:,i] != -999
		plt.scatter(tX[f, i], y[f])
		if saveDir != None:
			plt.savefig(saveDir + "/Feature" + str(i) + ".png")
		plt.show()


def dataHist(tX, y, saveDir=None):
	N = np.shape(tX)[1]
	for i in range(N):
		print("Feature " + str(i))
		f = tX[:,i] != -999
		b = y[:] == 1
		s = y[:] != 1
		plt.figure(figsize=(20,5))
		plt.subplot(121)
		plt.hist(tX[np.logical_and(f, b), i], 50, facecolor = "blue")
		plt.subplot(122)
		plt.hist(tX[np.logical_and(f, s), i], 50, facecolor = "red")
		if saveDir != None:
			plt.savefig(saveDir + "/Feature" + str(i) + ".png")	    
		plt.show()
