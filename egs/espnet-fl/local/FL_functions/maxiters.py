#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np

'''
   support script for I-Level strategy, we need to know max number of batches for training process
'''

def main():
    x=[]
    with open(args.infile,'r') as iterfile:
        for line in iterfile:
           val=int(line.rstrip())
           x.append(val)
    
    iterfile.close()
    with open(args.outfile,'w') as maxfile:
        print(max(x), file=maxfile)
    maxfile.close()    
    #return max(x)




def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--outfile", required=True, type=str)
    parser.add_argument('--infile', required=True, type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
