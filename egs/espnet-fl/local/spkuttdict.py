import os,sys
import json

def creatdict(srcdir, savedir):
    '''
    This code creat a dictionary, which indicate how many utterances has a speaker
    :param srcdir: The folder where saved feat.scp
    :param savedir: The folder save the results
    :return:
    '''
    savedir = os.path.join(savedir, 'train.json')
    dict = {}
    spklist = os.listdir(srcdir)
    for spk in spklist:
        utts = sum(1 for line in open(os.path.join(srcdir, spk, 'train', 'feats.scp')))
        dict.update({spk: utts})
    with open(savedir, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

creatdict(sys.argv[1],sys.argv[2])
