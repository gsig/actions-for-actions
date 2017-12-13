#!/usr/bin/env python
import numpy as np
import csv
import sys
import math
import combine_submissions
sys.path.insert(0, '../tool/') # for charades evaluator
from charades import LocalizationEvaluator
from charades import ClassificationEvaluator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['font.family']='CMU Serif'


######################################################
# Settings
GTFILE = '../tool/Charades_v1_test.csv'
ORACLES = [{'name':'Object',      'file':'oracle_object_charades.txt'   },
           {'name':'Verb',        'file':'oracle_verb_charades.txt'     },
           {'name':'Intent (30)', 'file':'oracle_intent30_charades.txt' },
           {'name':'Intent (50)', 'file':'oracle_intent50_charades.txt' },
           {'name':'Time',        'file':'oracle_time_charades.txt'     },
           {'name':'Pose (500)',  'file':'oracle_pose500_charades.txt'  },
          ]
Noracles = len(ORACLES)
submission_files = sys.argv[1:]
name = lambda x: x.split('/')[-1].split('.')[0]
seriesnames = [name(x) for x in submission_files]
Nsub = len(submission_files)
LEGEND = True
if Nsub==0: LEGEND = False


######################################################
# plot settings
plotname = 'output_oraclesplot_'+'_'.join([x.split('/')[-1].split('.')[0] for x in sys.argv[1:]])
XNAME = 'Average Precision'
YNAME = ''
COLORS = ['#88aec1','#765f50','#76cc88','#3f5d7d','#7d3f5d']
markers = ['s','o','^','D']
def setup_plot():
    fig = plt.figure(figsize=(5,2),facecolor='white')
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.ylabel(YNAME, fontsize=14)
    plt.xlabel(XNAME, fontsize=16)
    markers = ['s','o','^','D']
    return fig,ax

def finalize_plot(allticks,handles):
    plt.locator_params(axis='x', nticks=Noracles,nbins=Noracles)
    plt.yticks([x[0] for x in allticks], [x[1] for x in allticks])
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off'         # ticks along the top edge are off
    )
    if LEGEND:
        plt.legend([h[0] for h in handles],seriesnames,
                   loc='upper right',borderaxespad=0.,
                   ncol=1,fontsize=10,numpoints=1)
    plt.gcf().tight_layout()


######################################################
# Data processing 
def mAP(submission_file):
    # Calculate AP for each class
    # This version normalizes the AP for each class to the same number of positives and negatives. See paper
    print('Calculating mAP for {}'.format(submission_file))
    evaluator = ClassificationEvaluator(GTFILE, submission_file)
    _,_,APs = evaluator.evaluate_submission()
    return np.mean(APs)

def make_data():
    # if there are submission files combine them with oracles
    # otherwise just use the oracles
    newdata = [[] for _ in ORACLES]
    for i,oracle in enumerate(ORACLES):
        if Nsub==0:
            score = mAP(oracle['file'])
            newdata[i].append(score)
        else:
            for sub in submission_files:
                out = 'combinations/{}_{}.txt'.format(name(oracle['file']),name(sub))
                combine_submissions.main(oracle['file'],sub,out)
                print('Combination of {} and {} written to {}'.format(oracle['file'],sub,out))
                score = mAP(out)
                newdata[i].append(score)
    return newdata


######################################################
# Main function
def main():
    newdata = make_data()
    setup_plot()
    allticks = []
    for i,(oracle,data) in enumerate(zip(ORACLES,newdata)):
        pos = i*(max(1,Nsub)+1)+np.array(range(max(1,Nsub)+1))
        handles = []
        for j,dat in enumerate(data):
            h = plt.barh(pos[1]+j,dat,color=COLORS[j],edgecolor=COLORS[j])
            handles.append(h)
        allticks.extend(zip([x for x in pos], ['', oracle['name'],'']))
    finalize_plot(allticks,handles)
    plt.savefig(plotname+'.pdf')
    plt.show()
main()

