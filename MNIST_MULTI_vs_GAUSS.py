import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB


digits=load_digits()
X=digits.data
Y=digits.target
# print(X.shape)
# print(Y.shape)
# check=100
# print(Y[check])
# print(X[check])
# plt.imshow(X[check].reshape((8,8)),cmap='gray')
# plt.show()

mnb=MultinomialNB()
gnb=GaussianNB()

mnb.fit(X,Y)
gnb.fit(X,Y)
print(mnb.score(X,Y))
print(gnb.score(X,Y))

print(cross_val_score(mnb,X,Y,scoring='accuracy',cv=10).mean())
print(cross_val_score(gnb,X,Y,scoring='accuracy',cv=10).mean())




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



classes_labels = np.arange(10)

Y_mnb = mnb.predict(X)
cnf_matrix = confusion_matrix(Y,Y_mnb)
#print(cnf_matrix)

plot_confusion_matrix(cnf_matrix, classes=classes_labels,
                          normalize=False,
                          title='Confusion matrix Multinomial NB',
                          cmap=plt.cm.Accent)

Y_gnb = gnb.predict(X)
cnf_matrix = confusion_matrix(Y,Y_gnb)
#print(cnf_matrix)

plot_confusion_matrix(cnf_matrix, classes=classes_labels,
                          normalize=False,
                          title='Confusion matrix Gaussian NB',
                          cmap=plt.cm.Accent)


