#!/usr/bin/env python
#
# Script for combining submission files
#
# Contributor: Gunnar Atli Sigurdsson

import numpy as np
import sys

w = [0.5,0.5]
w = [x/sum(w) for x in w]
NCLASSES = 157

def loadfile(path):
    with open(path) as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    localization = len(lines[0]) == NCLASSES+2
    if localization:
        data = [(x[0]+' '+x[1],np.array([float(y) for y in x[2:]])) for x in lines]
    else:
        data = [(x[0],np.array([float(y) for y in x[1:]])) for x in lines]
    return data

def normme(x):
    x = x-np.mean(x)
    x = x/(0.00001+np.std(x))
    return x

def lookup(d,key):
    if key in d:
        return d[key]
    else:
        sys.stderr.write('error ' + key + '\n')
    return np.zeros((NCLASSES,))

def main(sub1file,sub2file,outfile):
    sub1 = loadfile(sub1file)
    sub2 = loadfile(sub2file)
    sub1dict = dict(sub1)
    sub2dict = dict(sub2)
    keys = list(set(sub1dict.keys()+sub2dict.keys()))
    with open(outfile,'w') as f:
        for id0 in keys:
            s1 = lookup(sub1dict,id0)
            s2 = lookup(sub2dict,id0)
            out = s1*w[0]+s2*w[1] #unnormalized combination
            #out = normme(s1)*w[0]+normme(s2)*w[1] #normalize first
            #out = np.exp(np.log(s1)*w[0]+np.log(s2)*w[1]) #weighted geometric mean
            out = [str(x) for x in out]
            f.write('{} {}\n'.format(id0,' '.join(out)))

if __name__ == '__main__':
    sub1file = sys.argv[1]
    sub2file = sys.argv[2]
    outfile = sys.argv[3]
    main(sub1file,sub2file,outfile)
