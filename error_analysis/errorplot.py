#!/usr/bin/env python
import numpy as np
import sys
import csv
subfile = sys.argv[1]
name = lambda x: x.split('/')[-1].split('.')[0]
plotname = 'output_errorplot_'+name(subfile)+'.pdf'
GTFILE = '../tool/Charades_v1_test.csv'
FPS = 24
det = [[] for _ in range(157)]
a_to_o,o_to_a,a_to_v,v_to_a = {},{},{},{} #mapping from action to object etc


######################################################
# plot settings
font = {'family' : 'CMU Serif',
        'weight' : 'normal',
        'size'   : 10}
import matplotlib
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
LABELS = ['TP', 'BND', 'OBJ', 'VRB', 'OTH', 'FP']
COLORS = ['#88aec1', '#765f50', '#76cc8e', '#3f5d7d', '#7d3f5d', '#808080']
def plot(sizes,plotname):
    fig = plt.figure(figsize=(2.0,2.0),facecolor='white')
    ax = plt.subplot(111)
    psizes = ['%1.1f%%' % (x/sum(sizes)*100) for x in sizes]
    labels = [x+'\n'+y for x,y in zip(LABELS,psizes)]
    patches = plt.pie(sizes, colors=COLORS, labels=labels, 
                      shadow=False, startangle=90, labeldistance=0.7, 
                      wedgeprops={'linewidth': 4})
    for pie_wedge in patches[0]:
        pie_wedge.set_edgecolor('white')
    for t in patches[1]:
        t.set_horizontalalignment('center')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(plotname)
    print('saved plot to {}'.format(plotname))
    plt.show()


######################################################
# Data processing 
s2c = lambda x: int(x[1:]) # string to class number
with open('Charades_v1_mapping.txt') as f:
    for line in f:
        x,y,z = line.strip().split()
        a,o,v = s2c(x),s2c(y),s2c(z)
        if not o in o_to_a: o_to_a[o] = []
        if not v in v_to_a: v_to_a[v] = []
        o_to_a[o].append(a)
        a_to_o[a] = o
        v_to_a[v].append(a)
        a_to_v[a] = v

def loadfile(path):
    with open(path) as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    arr = lambda x: np.array([float(y) for y in x])
    if len(lines[0]) == 158: #1 prediction per video 
        print('Warning: no localization submitted, replicating to all frames')
        return [(((l[0],i),arr(l[1:]))) for l in lines for i in range(1,26)]
    else:
        return [((l[0],int(l[1])),arr(l[2:])) for l in lines]

def parse_actions(actions):
    actions = [] if actions=='' else actions.split(';')
    actions = [x.split(' ') for x in actions]
    return [(s2c(x),float(y),float(z)) for x,y,z in actions]

def analyze(subdict,actions,aid,f,sec,fnum):
    # Assign the prediction score for each class in
    # each timepoint to a group if it is:
    #   tp: Class is present
    #   bnd: Class not present, but almost (timewise)
    #   fb: Class not present, but correct object and verb
    #   obj: Class not present, but correct object
    #   verb: Class not present, but correct verb
    #   fo: Class not present, neither object nor verb
    #   fp: No class present here
    # We will then sort these scores by magnitude and look
    # at the dsitribution of these labels among them
    frameacts = [a for a,s,e in actions if s<=sec<=e]
    framevrbs = [a_to_v[a] for a,s,e in actions if s<=sec<=e]
    frameobjs = [a_to_o[a] for a,s,e in actions if s<=sec<=e]
    framebnds = [a for a,s,e in actions if s-(e-s)/3.<=sec<=e+(e-s)/3.]
    for c in range(157):
        # for each class prediction at this timepoint
        if not (aid,f) in subdict: continue
        s = subdict[(aid,f)][c] # get the score for this class at this time
        if c in frameacts:
            # true, c class is present at this timepoint
            det[c].append((aid,f,s,fnum,'tp'))
        else:
            # false, c class is NOT present at this timepoint
            if c in framebnds:
                # almost though! Let's label this is boundary error
                det[c].append((aid,f,s,fnum,'bnd'))
            else:
                # Not even close, let's continue
                if len(frameacts) > 0:
                    # Are there any classes happening at this timepoint?
                    if a_to_o[c] in frameobjs and a_to_v[c] in framevrbs:
                        # got both object and verb correct!
                        det[c].append((aid,f,s,fnum,'fb',frameacts))
                    elif a_to_o[c] in frameobjs:
                        # got object correct!
                        det[c].append((aid,f,s,fnum,'obj',frameacts))
                    elif a_to_v[c] in framevrbs:
                        # got verb correct!
                        det[c].append((aid,f,s,fnum,'vrb',frameacts))
                    else:
                        # got nothing correct...
                        det[c].append((aid,f,s,fnum,'fo'))
                else:
                    # Nothing happening at this timepoint
                    det[c].append((aid,f,s,fnum,'fp'))


######################################################
# Main function
def main(subfile,plotname):
    # read in ground truth, and plot the distribution 
    # of the top scores for each class
    sub = loadfile(subfile)
    subdict = dict(sub)
    with open(GTFILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row['id']
            length = float(row['length'])
            Nframes = FPS*int(length)
            actions = parse_actions(row['actions'])
            frames = np.linspace(0,length,25)
            framenumber = np.linspace(0,Nframes,25)
            for f,sec,fnum in zip(range(1,26),frames,framenumber):
                analyze(subdict,actions,aid,f,sec,fnum)
    pies = []
    for c in range(157):
        scores = det[c]
        scores.sort(key=lambda x: -x[2])
        scores = scores[:100] # pick the top 100 scores for this class
        tp = sum(1 for x in scores if x[4]=='tp')
        bnd = sum(1 for x in scores if x[4]=='bnd')
        fb = sum(1 for x in scores if x[4]=='fb')
        obj = fb/2.+sum(1 for x in scores if x[4]=='obj')
        vrb = fb/2.+sum(1 for x in scores if x[4]=='vrb')
        fo = sum(1 for x in scores if x[4]=='fo')
        fp = sum(1 for x in scores if x[4]=='fp')
        pie = [tp, bnd, obj, vrb, fo, fp]
        pies.append(pie)
    sizes = [sum(x) for x in zip(*pies)] #combine results for all classes
    plot(sizes,plotname)

main(subfile,plotname)

