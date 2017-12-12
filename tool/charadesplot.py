#!/usr/bin/env python
import numpy as np
import csv
import sys
import math
from charades import LocalizationEvaluator
from charades import ClassificationEvaluator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['font.family']='CMU Serif'
GTFILE = 'Charades_v1_test.csv'


######################################################
# load attributes
def load(name):
    with open(name) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        attributes = dict([(x,{}) for x in header])
        for row in reader:
            rowid = row[header[0]]
            for x,d in row.iteritems():
                try: attributes[x][rowid]=float(d)
                except ValueError: attributes[x][rowid]=d
    return attributes
attributescls = load('Charades_v1_attributes_class.csv')
attributesvid = load('Charades_v1_attributes_video.csv')


######################################################
# plot settings
plotname = 'output_charadesplot_'+'_'.join([x.split('/')[-1].split('.')[0] for x in sys.argv[1:]])
seriesnames = [x.split('/')[-1].split('.')[0] for x in sys.argv[1:]]
ERRORBARS = ['none','standard','stddev','mad'][1] #choose error bars
YLIM = [0.05,0.35] #limits of the y axis
YNAME = 'Average Precision'
COLORS = ['#88aec1','#765f50','#76cc88','#3f5d7d','#7d3f5d']
MARKERS = ['s','o','^','D']
TICKNAMES = [None,[''],['N','Y'],['1','2','3'],['1','2','3','4'],['T','S','M','L','H'],['0','T','S','M','L','H'],['0','T','S','M','L','H','1'],['0','1','2','3','4','5','6','7']]
LEGEND = True
LABELVERSION = ['dual','single'][0] # how to arrange labels?
handles = []

# list of attributes, feel free to add your own, and add them to either attributescls or attributesvid
# Type: indicates if it is a class attribute or a video attribute, 
# Title: Name of the attribute (header in the attribte dictionaries)
# Groups: Center of each 'bin'. This was deterimined from clustering, feel free to modify.
# Header: Name of the plot section that starts with this attribute, see resulting plot for details.
series = [{'type':'class', 'title':'#Samples',    'groups':[100,400,800,1200],         'header':'Training Data',           },
          {'type':'class', 'title':'#Frames',     'groups':[1000,10000,20000],         'header':'',                        },
          {'type':'video', 'title':'Novel',       'groups':[-4.3,-1.33,1.67],          'header':'',                        },
          {'type':'video', 'title':'NewActor',    'groups':[100,0],                    'header':'',                        },
          {'type':'class', 'title':'Extent',      'groups':[10,15,20,25],              'header':'Temporal Reasoning',      },
          {'type':'video', 'title':'Extent-v',    'groups':[9.0,11.3,25.0],            'header':'',                        },
          {'type':'class', 'title':'Seq',         'groups':[0,1],                      'header':'',                        },
          {'type':'class', 'title':'Short',       'groups':[0,1],                      'header':'',                        },
          {'type':'class', 'title':'Passive',     'groups':[1,0],                      'header':'',                        },
          {'type':'class', 'title':'AvgMotion',   'groups':[.005,.0075,.01,.0125],     'header':'',                        },
          {'type':'video', 'title':'AvgMotion-v', 'groups':[0,.006,.017],              'header':'',                        },
          {'type':'video', 'title':'PersonSize',  'groups':[140,155,170],              'header':'Person-based\nReasoning', },
          {'type':'video', 'title':'People',      'groups':[1,2],                      'header':'',                        },
          {'type':'class', 'title':'PoseVar',     'groups':[.675, .725, .775, .825],   'header':'',                        },
          {'type':'class', 'title':'PoseRatio',   'groups':[0.4,0.5,0.6,0.7],          'header':'',                        },
          {'type':'class', 'title':'Obj',         'groups':[0,1],                      'header':'Fine-grained\nReasoning', },
          {'type':'class', 'title':'#Obj',        'groups':[1,4,8],                    'header':'',                        },
          {'type':'class', 'title':'#Verbs',      'groups':[1,5,10],                   'header':'',                        },
          {'type':'class', 'title':'Interact',    'groups':[0,1],                      'header':'',                        },
          {'type':'class', 'title':'Tool',        'groups':[0,1],                      'header':'',                        },
          {'type':'video', 'title':'#Actions',    'groups':[3,6,15],                   'header':'',                        },
         ]

def setup_plot(yname):
    # plot settings and font size
    fig = plt.figure(figsize=(10,3),facecolor='white')
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    plt.ylabel(yname, fontsize=14)
    return fig,ax

skips,headerloc,headers = {},[],[]
def update_locs(groups,j,count,header):
    # keeps track of what group should go where
    n = len(groups)
    if j not in skips:
        skips[j] = n
    else:
        n = skips[j]
    pos = count + np.array(range(n))
    count += n+1
    if not header=='':
        headerloc.append(pos[0])
        headers.append(header)
    return pos,count

def set_axes(ax,xticks,xlabels,xticks2,xlabels2,count,version):
    # fits all the x labels on the x axis by either rotating or stacking them
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    def printlabel(xticks2,xlabels2,ax,offset,rotation):
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", offset))
        ax2.set_frame_on(False)
        ax2.patch.set_visible(False)
        for sp in ax2.spines.itervalues():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(xticks2)
        ax2.set_xticklabels(xlabels2,verticalalignment='top',fontsize=12,rotation=rotation)
        ax2.xaxis.set_tick_params(width=0)
        ax2.set_xlim([0,count])
    if version=='dual': 
        printlabel(xticks2[::2],xlabels2[::2],ax,-0.15,0) # set 1st row label offset
        printlabel(xticks2[1::2],xlabels2[1::2],ax,-0.25,0) # set 2nd row label offset
    if version=='single':
        printlabel(xticks2,xlabels2,ax,-0.15,22) # set rotated label offset and rotation
    ax.set_xlim([0,count])

def finalize_plot(ax,fxticks,fxlabels,fxticks2,fxlabels2):
    # write labels and adjust layout
    count = fxticks[-1]+1
    set_axes(ax,fxticks,fxlabels,fxticks2,fxlabels2,count,version=LABELVERSION)
    plt.ylim(YLIM)
    for x,text in zip(headerloc,headers):
        ax.text(x-.5, YLIM[1], text, verticalalignment='top', horizontalalignment='left', fontsize=14,style='italic')
        plt.plot([x-1,x-1],[YLIM[0],YLIM[1]],
                 color='#000000', marker=None, 
                 linestyle="--",linewidth=1)
    if LEGEND:
        plt.legend([h[0] for h in handles],seriesnames,loc='upper center',borderaxespad=0.,
                   bbox_to_anchor=(0.5,-0.4),ncol=len(handles),fontsize=14,numpoints=1)
    plt.gcf().tight_layout()


######################################################
# Data processing 
def make_groups(groups,X,Y):
    # 'cluster' X based on the cluster bounds in 'groups' and report the statistics for the respective Ys
    newdata = []
    inv = groups[0]>groups[1]
    if inv: groups=groups[::-1]
    for s,g,e in zip([float('-inf')]+groups[:-1],groups,groups[1:]+[float('inf')]):
        s2,e2 = (s+g)/2,(e+g)/2
        dat = Y[(s2<X)&(X<=e2)]
        n = dat.shape[0]
        val = np.mean(dat) #np.median(dat)
        if ERRORBARS=='none': std = 0
        elif ERRORBARS=='std': std = np.std(dat)
        elif ERRORBARS=='standard': std = np.std(dat)/math.sqrt(n)
        elif ERRORBARS=='mad': std = np.median(np.abs(dat-np.median(dat)))
        newdata.append((g,val,std))
    if inv: newdata=newdata[::-1]
    return newdata

def classAP(submission_file,mask=None):
    # Calculate AP for each class
    # mask indicates if only certain videos should be used
    # This version normalizes the AP for each class to the same number of positives and negatives. See paper
    with open(submission_file) as f: N=len(f.readline().strip().split(' '))
    if N==157+2:
        evaluator = LocalizationEvaluator(GTFILE, submission_file, mask)
    else:
        evaluator = ClassificationEvaluator(GTFILE, submission_file, mask)
    _,_,APs = evaluator.evaluate_submission()
    score = dict([('c{:03d}'.format(i),x) for i,x in enumerate(APs)])
    return score

def subsetmAP(xdata,groups,submission_file):
    # calculate mAP over subset of videos
    newdata = []
    inv = groups[0]>groups[1]
    if inv: groups=groups[::-1]
    for s,g,e in zip([float('-inf')]+groups[:-1],groups,groups[1:]+[float('inf')]):
        s2,e2 = (s+g)/2,(e+g)/2
        mask = [x[0] for x in xdata.iteritems() if s2<x[1]<=e2]
        score = classAP(submission_file,mask)
        meanp = np.nanmean([x[1] for x in score.iteritems()])
        #stddev originally obtained with bootstrapping
        #omitted here for speed and replaced with a constant
        newdata.append((g,meanp,0.02)) 
    if inv: newdata=newdata[::-1]
    return newdata

scores = {}
def make_data(title,groups,attributetype,submission_file,baseline):
    # Attributes are either class based or video based
    # Calculate AP for each class and cluster those based on each class attribute
    # OR Cluster videos based on each video attribute, and then calculate mAP for each cluster
    newdata = []
    if attributetype == 'class':
        xdata = attributescls[title]
        if not submission_file in scores:
            scores[submission_file] = classAP(submission_file)
        score = scores[submission_file]
        data = []
        for x in sorted(score.keys()):
            data.append((xdata[x],score[x]))
        X = np.array([x[0] for x in data])
        Y = np.array([x[1] for x in data])
        newdata = make_groups(groups,X,Y)
    elif attributetype == 'video':
        xdata = attributesvid[title]
        newdata = subsetmAP(xdata,groups,submission_file)
        meanp = np.mean([x[1] for x in newdata])
        delta = meanp-baseline #fix mAP subset drift
        newdata = [(g,val-delta,std) for g,val,std in newdata]
    return newdata


######################################################
# Main function
def main():
    # loop over the submission file names in sys.argv
    # for each file, plot the attributes in 'series'
    fig,ax = setup_plot(YNAME)
    fxticks,fxlabels,fxticks2,fxlabels2 = [],[],[],[]
    for i,submission_file in enumerate(sys.argv[1:]):
        print('processing submission: {}'.format(submission_file))
        color = COLORS[i]
        score = classAP(submission_file)
        baseline = np.mean([x[1] for x in score.iteritems()])
        xticks,xlabels,xticks2,xlabels2,count = [],[],[],[],1
        for j,serie in enumerate(series):
            print('  processing attribute: {}'.format(serie['title']))
            title,groups,attributetype,header = serie['title'],serie['groups'],serie['type'],serie['header']
            newdata = make_data(title,groups,attributetype,submission_file,baseline)
            pos,count = update_locs(groups,j,count,header)
            if newdata==[]: continue
            plt.plot([pos[0],pos[-1]],[baseline]*2,color=color,marker=None,linestyle=":",linewidth=1)
            error = [x[2] for x in newdata]
            if ERRORBARS=='none': error=None
            h=plt.errorbar(pos, [x[1] for x in newdata],color=color,marker=None, 
                         linestyle="-",linewidth=3,yerr=error,clip_on=False)
            xticks += [x for x in pos]
            xlabels += TICKNAMES[len(groups)]
            xticks2 += [(pos[0]+pos[-1])/2.]
            xlabels2 += [title]
        if i==0: fxticks,fxlabels,fxticks2,fxlabels2 = xticks,xlabels,xticks2,xlabels2
        handles.append(h)
    finalize_plot(ax,fxticks,fxlabels,fxticks2,fxlabels2)
    plt.savefig(plotname+'.pdf',bbox_inches='tight')
    plt.show()
main()

