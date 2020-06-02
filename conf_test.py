#%%
import matplotlib.pyplot as plt
import numpy as np

def plotConfMatrix(confMatrix, labels):
# def plotConfMatrix(self, confMatrix, labels, plotpath):
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.set_aspect(1)
      cax = ax.matshow(confMatrix)

      for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
          c = confMatrix[j,i]
          if c > (0.75*np.max(confMatrix)):
              color = 'black'
          else:
              color = 'w'
          ax.text(i, j, str(np.round(c,2)),color=color, va='center', ha='center')

      plt.title('Confusion matrix')
      fig.colorbar(cax)
      ax.set_xticks(range(len(labels)))
      ax.set_yticks(range(len(labels)))
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)
      ax.set_ylim(len(labels)-0.5, 0-0.5)

      plt.xlabel('Predicted')
      plt.ylabel('True')

    #   plt.savefig(plotpath)

# %%
