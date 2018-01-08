from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
from scipy import stats

#=================FREQUENCY OF EMOTIONS=================

file_names = ['output_mouth_2.txt', 'output_eyes_2.txt']
emotion_files = ['emotions_10k_0_3388.txt', 'emotions_10k_3389_6778.txt', 'emotions_10k_6779_10167.txt', 'emotions_jaffe.txt']

emotions = []
for file in emotion_files:
    f = open(file, 'r+')
    emotions.extend(map(int, f.read().split('\n')))

count = Counter(emotions)

plt.hist(emotions)
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.savefig('frequencies.png', pad_inches=0.1, bbox_inches='tight')

#=================TRAINING ACCURACY=================
for fname in file_names:
    #   Clearing the Plot
    plt.clf()
    f = file(fname)
    lines = f.read().split('\n')
    i = 0
    training_accuracies = []
    #   Extracting Training Accuracies
    while i < len(lines) and 'Average Accuracy' not in lines[i]:
        if 'Minibatch accuracy' in lines[i]:
            training_accuracies.append(float(lines[i][20:lines[i].index('%', 20)]))
        i += 1

    train_accuracies = []
    #   Scaling down to fit the graph in a compact image
    for j in xrange(len(training_accuracies)):
        if (j * 50) % 100 == 0:
            train_accuracies.append(training_accuracies[j]) 
    #   Plotting Training Accuracies
    temp = 0
    if len(train_accuracies) == 100:
        temp = 100
    else:
        temp = 50
    plt.plot(range(temp), train_accuracies, color='blue', label='Training')
    #   Calculating Training Mode
    train_mode = (stats.mode(train_accuracies).mode[0])
    i += 1
    
    #   Formatting and Saving the Graph
    plt.title(fname + '\nTraining Mode: ' + str(train_mode))
    plt.ylabel('Percentage Accuracy')
    plt.xlabel('Number of iterations(x100)')
    plt.legend()
    figname = fname[0:fname.index('.')] + '.png'
    plt.savefig(figname, pad_inches=0.1, bbox_inches='tight')

#=================INFORMATION LOSS=================
for fname in file_names:
    plt.clf()
    f = file(fname)
    lines = f.read().split('\n')

    i = 0

    #   Extracting Information Loss
    info_loss_training = []
    while i < len(lines) and 'Average Accuracy' not in lines[i]:
        if 'Minibatch loss' in lines[i]:
            info_loss_training.append(float(lines[i][lines[i].index(':') + 1:].strip()))
        i += 1
    info_loss_train = []
    #   Scaling down to fit the graph
    for j in xrange(len(info_loss_training)):
        if (j * 50) % 100 == 0:
            info_loss_train.append(info_loss_training[j]) 
    #   Plotting the Training Data
    if len(info_loss_train) == 100:
        temp = 100
    else:
        temp = 50
    plt.plot(range(temp), info_loss_train, color='blue', label='Training')
    i += 1

    #   Saving the Graph
    plt.title(fname + '\nInformation Loss')
    plt.ylabel('Information loss')
    plt.xlabel('Number of iterations(x100)')
    plt.legend()
    figname = fname[0:fname.index('.')] + '_loss.png'
    plt.savefig(figname, pad_inches=0.1, bbox_inches='tight')