##########################################################
# This file generates the sets for training and testing.
#
# Ruhr-University Bochum
# Cognitive Signal Processing Group
# April 2021
##########################################################

import sys
import os
import json
import shutil
import os
import copy



def presplit_data(spkuttdict, threshold):
    '''
    pre-split the speaker from the spkuttdict by using threshold
    Args:
        spkuttdict: the dictionary  saves the information about speaker and Utts
        threshold: the speaker has more Utts than threshold, will be treat as FL User

    Returns: fldict, pretraindict: the dictionaries. the key is the speaker id and the values are the Utt ids

    '''
    fldict = {}
    pretraindict = {}
    spkkeys = spkuttdict.keys()
    for spkkey in spkkeys:
        if len(spkuttdict[spkkey]) >= threshold:
            fldict.update({spkkey: spkuttdict[spkkey]})
        else:
            pretraindict.update({spkkey: spkuttdict[spkkey]})
    return fldict, pretraindict


def split_data(spkuttdict, sourcedir, savedir):
    '''
    split the dataset
    Args:
        spkuttdict: a dictionary, save the information of different dataset
        sourcedir: the dir where saves the kaldi files
        savedir: the dir save the splited data

    Returns:

    '''

    for dset in ["train_960_org", "dev_org", "test_org"]:
        textdict = {}
        wavdict = {}
        '''if len(spkuttdict) == 3:
            if dset == "test_org":  ## prepare the text and wav file, because test set has a part from train set
                with open(os.path.join(sourcedir, "train_960_org", 'text')) as f:  ## read the text file
                    lines = f.read().splitlines()
                with open(os.path.join(sourcedir, dset, 'text')) as f:  ## read the
                    testlines = f.read().splitlines()
                lines.extend(testlines)
                for line in lines:
                    splitlines = line.split(' ')
                    textdict.update({splitlines[0]: ' '.join(splitlines[1:])})

                with open(os.path.join(sourcedir, "train_960_org", 'wav.scp')) as f:
                    lines = f.read().splitlines()
                with open(os.path.join(sourcedir, dset, 'wav.scp')) as f:
                    testlines = f.read().splitlines()
                lines.extend(testlines)
                for line in lines:
                    splitlines = line.split(' ')
                    wavdict.update({splitlines[0]: ' '.join(splitlines[1:])})

            else:
                with open(os.path.join(sourcedir, dset, 'text')) as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        splitlines = line.split(' ')
                        textdict.update({splitlines[0]: ' '.join(splitlines[1:])})

                with open(os.path.join(sourcedir, dset, 'wav.scp')) as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        splitlines = line.split(' ')
                        wavdict.update({splitlines[0]: ' '.join(splitlines[1:])})
        else:'''
        with open(os.path.join(sourcedir, dset, 'text')) as f:
            lines = f.read().splitlines()
            for line in lines:
                splitlines = line.split(' ')
                textdict.update({splitlines[0]: ' '.join(splitlines[1:])})

        with open(os.path.join(sourcedir, dset, 'wav.scp')) as f:
            lines = f.read().splitlines()
            for line in lines:
                splitlines = line.split(' ')
                wavdict.update({splitlines[0]: ' '.join(splitlines[1:])})


        values = spkuttdict[dset][1].values()  ## the pretrain set Utts
        values = list(values)
        pretrainlist = []  ## here saved all Utts in pretrain set
        for i in range(len(values)):
            pretrainlist.extend(values[i])

        #### first split pretrain dset.
        pretraintext = []
        pretrainwav = []
        pretrainuts = []
        for pspk in pretrainlist:  ## make pretrain set text, wav and utt2spk files
            pretraintext.append(pspk + ' ' + textdict[pspk] + '\n')
            pretrainwav.append(pspk + ' ' + wavdict[pspk] + '\n')
            pretrainuts.append(pspk + ' ' + pspk.split('-')[0] + '\n')
        savepretraindir = os.path.join(savedir, "pretrain", dset.split('_')[0])
        if not os.path.exists(os.path.join(savepretraindir)):
            os.makedirs(os.path.join(savepretraindir))
        write(savepretraindir, pretrainuts, pretraintext, pretrainwav)

        if len(spkuttdict) > 3:  ## split the test Utts, which are from the pretrain train set
            if dset =="train_960_org":
                for transfset in ['transfer_fl_dev_org', 'transfer_fl_test_org', 'transfer_pre_dev_org', 'transfer_pre_test_org']:
                    values = spkuttdict[transfset].values()  ## the pretrain set Utts
                    values = list(values)
                    transferlist = []  ## here saved all Utts in pretrain set
                    for i in range(len(values)):
                        a = values[i]
                    transferlist = values
                    transftext = []
                    transfwav = []
                    transfuts = []
                    for pspk in transferlist:  ## make pretrain set text, wav and utt2spk files
                        transftext.append(pspk + ' ' + textdict[pspk] + '\n')
                        transfwav.append(pspk + ' ' + wavdict[pspk] + '\n')
                        transfuts.append(pspk + ' ' + pspk.split('-')[0] + '\n')
                    savetransferdir = os.path.join(savedir, "pretrain", transfset)
                    if not os.path.exists(os.path.join(savetransferdir)):
                        os.makedirs(os.path.join(savetransferdir))
                    write(savetransferdir, transfuts, transftext, transfwav)



        ## here split fl speakers
        flspks = spkuttdict[dset][0].keys()
        for flspk in flspks:
            savetraindir = os.path.join(savedir, "FL", flspk)
            if not os.path.exists(os.path.join(savetraindir)):  ## make subfolder for the speaker
                os.makedirs(os.path.join(savetraindir))
            splitfldata(textdict, wavdict, spkuttdict[dset][0][flspk], savetraindir)


def make_new_dataset(sourcedir, savedir, movedlines):
    traintextdict = {}
    testtextdict = {}
    with open(os.path.join(sourcedir, "train_960_org", 'text')) as f:
        trainlines = f.read().splitlines()
        for line in trainlines:
            splitlines = line.split(' ')
            traintextdict.update({splitlines[0]: ' '.join(splitlines[1:])})
    with open(os.path.join(sourcedir, "test_org", 'text')) as f:
        testlines = f.read().splitlines()
        for line in testlines:
            splitlines = line.split(' ')
            testtextdict.update({splitlines[0]: ' '.join(splitlines[1:])})
    trainwavdict = {}
    testwavdict = {}
    with open(os.path.join(sourcedir, "train_960_org", 'wav.scp')) as f:
        trainlines = f.read().splitlines()
        for line in trainlines:
            splitlines = line.split(' ')
            trainwavdict.update({splitlines[0]: ' '.join(splitlines[1:])})
    with open(os.path.join(sourcedir, "test_org", 'wav.scp')) as f:
        testlines = f.read().splitlines()
        for line in testlines:
            splitlines = line.split(' ')
            testwavdict.update({splitlines[0]: ' '.join(splitlines[1:])})
    values = movedlines.values()
    values = list(values)
    movedlist = []
    for i in range(len(values)):
        movedlist.extend(values[i])

    trainkeys = copy.deepcopy(list(traintextdict.keys()))
    for key in trainkeys:
        if key in movedlist:
            testtextdict.update({key: traintextdict[key]})
            testwavdict.update({key: trainwavdict[key]})
            del trainwavdict[key]
            del traintextdict[key]

    testkeys = list(testtextdict.keys())
    #### first split train.
    testtext = []
    testwav = []
    testuts = []
    for pspk in testkeys:
        testtext.append(pspk + ' ' + testtextdict[pspk] + '\n')
        testwav.append(pspk + ' ' + testwavdict[pspk] + '\n')
        testuts.append(pspk + ' ' + pspk.split('-')[0] + '\n')
    savepretraindir = os.path.join(savedir, "test_org")
    if not os.path.exists(os.path.join(savepretraindir)):
        os.makedirs(os.path.join(savepretraindir))
    write(savepretraindir, testuts, testtext, testwav)


    trainkeys = list(traintextdict.keys())
    #### first split train.
    traintext = []
    trainwav = []
    trainuts = []
    for pspk in trainkeys:
        traintext.append(pspk + ' ' + traintextdict[pspk] + '\n')
        trainwav.append(pspk + ' ' + trainwavdict[pspk] + '\n')
        trainuts.append(pspk + ' ' + pspk.split('-')[0] + '\n')
    savepretraindir = os.path.join(savedir, "train_960_org")
    if not os.path.exists(os.path.join(savepretraindir)):
        os.makedirs(os.path.join(savepretraindir))
    write(savepretraindir, trainuts, traintext, trainwav)


def splitfldata(textdict, wavdict, fllist, savetraindir):
    '''
    split the Fl speakers, 60% are in fl train set, 20% in fl dev set and 20% in fl test set
    Args:
        textdict: kaldi text file
        wavdict: kaldi wav file
        fllist: the utts from the speakers
        savetraindir: where save the data

    Returns:

    '''
    trainuts = []
    traintext = []
    trainwav = []
    devuts = []
    devtext = []
    devwav = []
    evaluts = []
    evaltext = []
    ealwav = []
    length = len(fllist)
    trainspk = fllist[: int(0.6 * length)]  ## 60% in train set
    devspk = fllist[int(0.6 * length): int(0.8 * length)]  ## 20% in dev set
    evalspk = fllist[int(0.8 * length) :]  ## 20% in test set

    for spk in trainspk:
        traintext.append(spk + ' ' + textdict[spk] + '\n')
        trainwav.append(spk + ' ' + wavdict[spk] + '\n')
        trainuts.append(spk + ' ' + spk.split('-')[0] + '\n')
    for spk in devspk:
        devtext.append(spk + ' ' + textdict[spk] + '\n')
        devwav.append(spk + ' ' + wavdict[spk] + '\n')
        devuts.append(spk + ' ' + spk.split('-')[0] + '\n')
    for spk in evalspk:
        evaltext.append(spk + ' ' + textdict[spk] + '\n')
        ealwav.append(spk + ' ' + wavdict[spk] + '\n')
        evaluts.append(spk + ' ' + spk.split('-')[0] + '\n')
    write(os.path.join(savetraindir, "dev"), devuts, devtext, devwav)
    write(os.path.join(savetraindir, "test"), evaluts, evaltext, ealwav)
    write(os.path.join(savetraindir, "train"), trainuts, traintext, trainwav)


def write(savedir, uts, text, wav):
    if not os.path.exists(os.path.join(savedir)):
        os.makedirs(os.path.join(savedir))
    uts.sort()
    text.sort()
    wav.sort()
    with open(savedir + '/utt2spk', 'at') as af:
        af.writelines(uts)
        af.close()
    with open(savedir + '/text', 'at') as af:
        af.writelines(text)
        af.close()
    with open(savedir + '/wav.scp', 'at') as af:
        af.writelines(wav)
        af.close()


def get_speech_num(spk2uttdir):
    '''
    This function use spk2utt get how many Utts in each speaker
    Args:
        spk2uttdir: the dir save spk2utt file

    Returns: spkdict: a dictionary, key is each speaker id, value is the Utts number for each speaker

    '''
    with open(spk2uttdir, "r") as text_file:  ## open this file
        lines = text_file.readlines()   ## read file into python and save in list
        tempdict = {}  ## initial a dict, to save the number of speech for each speakers.
        speakerlist = []
        spkdict = {}
        for i in range(len(lines)):
            lines[i] = lines[i].split(" ")
            speakerlist.append(lines[i][0].split("-")[0])
        for i in speakerlist:
            tempdict.update({i: {}})
        for i in range(len(lines)):
            speakerid = lines[i][0].split("-")[0]
            tempdict[speakerid].update({lines[i][0]: len(lines[i]) - 1})
        for i in speakerlist:
            sums = 0
            subkey = tempdict[i].keys()
            for key in subkey:
                sums = sums + tempdict[i][key]
            spkdict.update({i: sums})
    return spkdict


def get_gender_info(spk2genderdir):
    '''
    This function get gender information from the spk2gender file
    Args:
        spk2genderdir: the dir, where save the speaker gender information

    Returns: genderdict: a dictionary, key is "m" -> male and "f" -> female, the value is the speaker id

    '''
    with open(spk2genderdir, "r") as text_file:  ## open this file
        lines = text_file.readlines()   ## read file into python and save in list
        genderdict = {}  ## initial a dict, to save the number of speech for each speakers.
        malelist = []
        femallist = []
        for i in lines:
            if "m" in i:
                spkid = i.split("-")[0]
                malelist.append(spkid)
            elif "f" in i:
                spkid = i.split("-")[0]
                femallist.append(spkid)
        genderdict.update({"m": list(set(malelist))})
        genderdict.update({"f": list(set(femallist))})
    return genderdict


def speaker_utt_dict(textdir):
    '''
    get the dictionary from text file, it saves the information about speaker and Utts
    Args:
        textdir: the text file from kaldi

    Returns: dicts is a dictionary, key is the speaker id, value is the coresponding Utts id

    '''
    with open(textdir, "r") as text_file:  ## open this file
        lines = text_file.readlines()  ## read file into python and save in list
        dicts = {}
        spkerlist = []
        for i in lines:
            spkerlist.append(i.split(" ")[0].split("-")[0])
        spkerlist = list(set(spkerlist))
        for i in spkerlist:
            dicts.update({i: []})
        for i in lines:
            spkuttid = i.split(" ")[0]
            spkerid = spkuttid.split("-")[0]
            dicts[spkerid].append(spkuttid)
        return dicts

if __name__ == '__main__':
    stage = 0  # the script should always run from stage 0

    #sourcedir = "./librispeech/data"  ##data file, which is generated by kaldi (espnet)
    #dsets = os.listdir(sourcedir)  ##the dataset name
    #savedir = './processing'
    #combinetest = False ## we move a part of pretrain train set to pretrain test set. "True" means we combine
                       ## the pretrain test set with Utts from pretrain train set. Otherwise they are in
                       ## different set

    sourcedir = sys.argv[1]  ##data file, which is generated by kaldi (espnet)
    dsets = list(filter(lambda x: '_org' in x, os.listdir(sourcedir)))  ##the dataset name
    savedir = sys.argv[2]

    if stage <= 0:  ## make gegnderdict
        gendersavedir = os.path.join(savedir, "genderdict") ## save genderdict in this folder
        if not os.path.exists(os.path.join(gendersavedir)):
            os.makedirs(os.path.join(gendersavedir))  ## make the genderdict folder
        for dset in dsets: ##dsets:[train_960_org, dev_org, test_org]
            spk2uttdir = os.path.join(sourcedir, dset, "spk2utt")
            spk2genderdir = os.path.join(sourcedir, dset, "spk2gender")
            spkdict = get_speech_num(spk2uttdir)  ## get how many Utts in each speaker
            spkgenderdict = get_gender_info(spk2genderdir)  ## get the speaker gender information
            outdict = {}
            outdict.update({"m": {}})
            outdict.update({"f": {}})
            malelist = spkgenderdict["m"]
            femalelist = spkgenderdict["f"]
            for m in malelist:   ## split male speakers
                outdict["m"].update({m: spkdict[m]})
            outdict["m"] = dict(sorted(outdict["m"].items(), key=lambda item: item[1]))
            for m in femalelist:    ## split female speakers
                outdict["f"].update({m: spkdict[m]})
            outdict["f"] = dict(sorted(outdict["f"].items(), key=lambda item: item[1]))
            ## save gender dictionary
            with open(os.path.join(gendersavedir, dset + "statistc.json"), 'w', encoding='utf-8') as f:
                json.dump(outdict, f, ensure_ascii=False, indent=4)

    if stage <= 1:  ## precheck dataset
        ## first check if dev and test set speaker in train set
        gendersavedir = os.path.join(savedir, "genderdict")
        with open(os.path.join(gendersavedir, "train_960_orgstatistc.json"), "r") as json_file:
            train = json.load(json_file)
        with open(os.path.join(gendersavedir, "test_orgstatistc.json"), "r") as json_file:
            test = json.load(json_file)
        with open(os.path.join(gendersavedir, "dev_orgstatistc.json"), "r") as json_file:
            dev = json.load(json_file)
        maletrain = list(train["m"].keys())   ## list of male speakers in train set
        femaletrain = list(train["f"].keys())  ## list of female speakers in train set
        maletest = list(test["m"].keys())  ## list of male speakers in test set
        femaletest = list(test["f"].keys())  ## list of female speakers in test set
        maledev = list(dev["m"].keys())  ## list of male speakers in dev set
        femaledev = list(dev["f"].keys())  ## list of female speakers in dev set

        testsymbol = 0
        devsymbol = 0
        for mltest in maletest:
            if mltest in maletrain:  ## if male speaker from the test set also in train set
                testsymbol = testsymbol + 1
        for fmtest in femaletest:  ## if female speaker from the test set also in train set
            if fmtest in femaletrain:
                testsymbol = testsymbol + 1
        for mldev in maledev:  ## if male speaker from the dev set also in train set
            if mldev in maletrain:
                devsymbol = devsymbol + 1
        for fmdev in femaledev:  ## if female speaker from the dev set also in train set
            if fmdev in femaletrain:
                devsymbol = devsymbol + 1
        if testsymbol == 0:
            print("test speakers are not in train set")
        if devsymbol == 0:
            print("dev speakers are not in train set")

        for dset in dsets:
            with open(os.path.join(gendersavedir, dset + "statistc.json"), "r") as json_file:  ## reload the gender information
                lines = json.load(json_file)
            pretrainmalesum = 0  ## how many Utts in pretrain male set
            trainmalesum = 0  ## how many Utts in train male set
            pretrainfemalesum = 0  ## how many Utts in pretrain female set
            trainfemalesum = 0    ## how many Utts in train female set
            malekeys = list(lines["m"].keys())  ## get the list of the male speaker list
            malekeys.reverse()  ## reverse the male speaker list, because we want from the speaker, who has
                                ## largest Utts as the FL speaker
            femalekeys = list(lines["f"].keys())
            femalekeys.reverse()  ## reverse the female speaker list, because we want from the speaker, who has
                                  ## largest Utts as the FL speaker
            malesum = 0  ## how many Utts for male speakers
            femalesum = 0  ## how many Utts for female speakers
            for mk in malekeys:
                malesum = malesum + lines["m"][mk]
            for fk in femalekeys:
                femalesum = femalesum + lines["f"][fk]

            i = 0
            while trainmalesum / malesum <= 0.65:  ## Accumulate Utts of male speakers, until it
                                                   ## has 65 % in FL set
                trainmalesum = trainmalesum + lines["m"][malekeys[i]]
                i = i + 1
            print(lines["m"][malekeys[i - 1]])  ## print the threshold of the male speaker
            for j in range(i, len(malekeys)):
                pretrainmalesum = pretrainmalesum + lines["m"][malekeys[j]]

            i = 0
            while trainfemalesum / femalesum <= 0.65:  ## Accumulate Utts of female speakers, until it
                                                       ## has 65 % in FL set
                trainfemalesum = trainfemalesum + lines["f"][femalekeys[i]]
                i = i + 1
            print(lines["f"][femalekeys[i - 1]])  ## print the threshold of the female speaker
            for j in range(i, len(femalekeys)):
                pretrainfemalesum = pretrainfemalesum + lines["f"][femalekeys[j]]

            print(dset + 'pretrainmale: %.f ' % pretrainmalesum)
            print(dset + 'fltrainmale: %.f ' % trainmalesum)
            print(dset + 'pretrainfemale: %.f ' % pretrainfemalesum)
            print(dset + 'fltrainfemale: %.f ' % trainfemalesum)

    threshold = 116   ## we find if we take the speakers, who has more than 116 Utts as Fl speaker,
                      ## FL dataset has 60% than pretrain train set.
    if stage <= 2:  ## split dataset
        spkdicts = {}
        print(dsets)
        for dset in dsets:
            splitsavedir = os.path.join(savedir, "data")
            textdir = os.path.join(sourcedir, dset, "text")
            spkuttdict = speaker_utt_dict(textdir)
            outdictlist = presplit_data(spkuttdict, threshold)
            spkdicts.update({dset: outdictlist})

        FLspeakerkeys = spkdicts["train_960_org"][0].keys()    ## the FL speaker list
        pretraintrainkeys = spkdicts["train_960_org"][1].keys()   ## the pretrain speaker list
        transferfldict = {}
        transferpretraindict = {}
        for pttkey in pretraintrainkeys:  ## get one utt from every pretrain speaker, so there are 974 Utts. one half is in pretrain dev and one half is in pretrain test
            transferpretraindict.update({pttkey: spkdicts["train_960_org"][1][pttkey][0]})
            del spkdicts["train_960_org"][1][pttkey][0]
        for fltkey in FLspeakerkeys:   ## get one utt from every FL speaker, so there are 1364 Utts. one half is in pretrain dev and one half is in pretrain test
            transferfldict.update({fltkey: spkdicts["train_960_org"][0][fltkey][0]})
            del spkdicts["train_960_org"][0][fltkey][0]

        pretraintrainspkuttnum = int(37129 / len(pretraintrainkeys)) + 1
        pretraindevspkuttnum = int(500 / len(spkdicts["dev_org"][1].keys()))
        pretraintestspkuttnum = int(500 / len(spkdicts["test_org"][1].keys()))

        pretraintraindict = {}
        for key in pretraintrainkeys:
            pretraintraindict.update({key: spkdicts["train_960_org"][1][key][: pretraintrainspkuttnum]})
        spkdicts["train_960_org"][1].update(pretraintraindict)

        pretraindevdict = {}
        for key in spkdicts["dev_org"][1].keys():
            pretraindevdict.update({key: spkdicts["dev_org"][1][key][: pretraindevspkuttnum]})
        spkdicts["dev_org"][1].update(pretraindevdict)

        pretraintestdict = {}
        for key in spkdicts["test_org"][1].keys():
            pretraintestdict.update({key: spkdicts["test_org"][1][key][: pretraintestspkuttnum]})
        spkdicts["test_org"][1].update(pretraintestdict)

        ## to ensure we generate same dataset, when we run the code, we have to sort the keys of transferfldict and transferpretraindict
        transfflkeys = list(transferfldict.keys())    ## the FL speaker list
        transfflkeys.sort()
        transfpretrainkeys = list(transferpretraindict.keys())
        transfpretrainkeys.sort()
        transffldevlist = transfflkeys[: int(len(transfflkeys)/2)]
        transffltestlist = transfflkeys[int(len(transfflkeys)/2) :]
        transfpretraindevlist = transfpretrainkeys[: int(len(transfpretrainkeys)/2)]
        transfpretraintestlist = transfpretrainkeys[int(len(transfpretrainkeys)/2) :]
        transferfldevdict = {}
        for key in transffldevlist:
            transferfldevdict.update({key: transferfldict[key]})
        spkdicts.update({"transfer_fl_dev_org": transferfldevdict})
        transferfltestdict = {}
        for key in transffltestlist:
            transferfltestdict.update({key: transferfldict[key]})
        spkdicts.update({"transfer_fl_test_org": transferfltestdict})
        transferpredevdict = {}
        for key in transfpretraindevlist:
            transferpredevdict.update({key: transferpretraindict[key]})
        spkdicts.update({"transfer_pre_dev_org": transferpredevdict})
        transferpretestdict = {}
        for key in transfpretraintestlist:
            transferpretestdict.update({key: transferpretraindict[key]})
        spkdicts.update({"transfer_pre_test_org": transferpretestdict})

        '''if combinetest is True:
            spkdicts["test_org"][1].update(transferdict) ## move 2 Utts from each speaker in pretrain set to the
                                                     ## pretrain test set
        else:
            spkdicts.update({"transfer_org": transferdict})

        with open(os.path.join(savedir, "from_train_to_test.json"), 'w', encoding='utf-8') as f:
            json.dump(transferdict, f, ensure_ascii=False, indent=4)'''

        split_data(spkdicts, sourcedir, splitsavedir)

    if stage <= 3:  ## check statistic
        # first check how many Utts in each set
        pretraintrain = 0
        pretraindev = 0
        pretraintest = 0
        fltrain = 0
        fldev = 0
        fltest = 0
        splitsavedir = os.path.join(savedir, "data")
        pretraintrain = len(open(os.path.join(savedir, "data", "pretrain", "train", "text")).readlines())
        pretraindev = len(open(os.path.join(savedir, "data", "pretrain", "dev", "text")).readlines())
        pretraintest = len(open(os.path.join(savedir, "data", "pretrain", "test", "text")).readlines())
        pretraintransfldev = len(open(os.path.join(savedir, "data", "pretrain", "transfer_fl_dev_org", "text")).readlines())
        pretraintransfltest = len(open(os.path.join(savedir, "data", "pretrain", "transfer_fl_test_org", "text")).readlines())
        pretraintrandpredev = len(open(os.path.join(savedir, "data", "pretrain", "transfer_pre_dev_org", "text")).readlines())
        pretraintrandpretest = len(
            open(os.path.join(savedir, "data", "pretrain", "transfer_pre_test_org", "text")).readlines())
        flsavedir = os.path.join(savedir, "data", "FL")
        spkerlist = os.listdir(flsavedir)
        for spker in spkerlist:
            fltrain = fltrain + len(open(os.path.join(flsavedir, spker, "train", "text")).readlines())
            fldev = fldev + len(open(os.path.join(flsavedir, spker, "dev", "text")).readlines())
            fltest = fltest + len(open(os.path.join(flsavedir, spker, "test", "text")).readlines())

        print('pretrain train set has %.f Utts' % pretraintrain)
        print('pretrain test set has %.f Utts' % pretraintest)
        print('pretrain dev set has %.f Utts' % pretraindev)
        print('pretrain transfer fl dev set has %.f Utts' % pretraintransfldev)
        print('pretrain transfer fl test set has %.f Utts' % pretraintransfltest)
        print('pretrain transfer pretrain-train  dev set has %.f Utts' % pretraintrandpredev)
        print('pretrain transfer pretrain-train  test set has %.f Utts' % pretraintrandpretest)
        print('FL train set has %.f Utts' % fltrain)
        print('FL test set has %.f Utts' % fltest)
        print('FL dev set has %.f Utts' % fldev)
        maleflspker = 0
        femaleflspker = 0

        males = []
        females = []
        for dset in dsets:
            gendersavedir = os.path.join(savedir, "genderdict")
            with open(os.path.join(gendersavedir, dset + "statistc.json"), "r") as json_file:
                lines = json.load(json_file)
            males.extend(lines['m'].keys())
            females.extend(lines['f'].keys())
        fluttsdict = {}
        for dset in dsets:
            spkerlist = os.listdir(flsavedir)
            malenum = 0
            femalenum = 0
            for i in spkerlist:
                if i in males:
                    malenum = malenum + len(open(os.path.join(flsavedir, i, dset.split("_")[0], "text")).readlines())
                else:
                    femalenum = femalenum + len(
                        open(os.path.join(flsavedir, i, dset.split("_")[0], "text")).readlines())
            fluttsdict.update({dset.split("_")[0] + "male": malenum})
            fluttsdict.update({dset.split("_")[0] + "female": femalenum})

        pretrainuttsdict = {}
        for dset in dsets:
            malenum = 0
            femalenum = 0
            with open(os.path.join(savedir, "data", "pretrain", dset.split("_")[0], 'text')) as f:
                lines = f.read().splitlines()
            for utts in lines:
                if utts.split("-")[0] in males:
                    malenum = malenum + 1
                else:
                    femalenum = femalenum + 1
            pretrainuttsdict.update({dset.split("_")[0] + "male": malenum})
            pretrainuttsdict.update({dset.split("_")[0] + "female": femalenum})

        for dset in ['transfer_fl_dev_org', 'transfer_fl_test_org', 'transfer_pre_dev_org', 'transfer_pre_test_org']:
            malenum = 0
            femalenum = 0
            with open(os.path.join(savedir, "data", "pretrain", dset, 'text')) as f:
                lines = f.read().splitlines()
            for utts in lines:
                if utts.split("-")[0] in males:
                    malenum = malenum + 1
                else:
                    femalenum = femalenum + 1
            pretrainuttsdict.update({dset + "male": malenum})
            pretrainuttsdict.update({dset + "female": femalenum})
        print('pretrain train set males have %.f Utts' % pretrainuttsdict["trainmale"])
        print('pretrain train set females have %.f Utts' % pretrainuttsdict["trainfemale"])
        print('pretrain test set males have %.f Utts' % pretrainuttsdict["testmale"])
        print('pretrain test set females have %.f Utts' % pretrainuttsdict["testfemale"])
        print('pretrain dev set males have %.f Utts' % pretrainuttsdict["devmale"])
        print('pretrain dev set females have %.f Utts' % pretrainuttsdict["devfemale"])

        print('pretrain transfer fl dev set males have %.f Utts' % pretrainuttsdict["transfer_fl_dev_orgmale"])
        print('pretrain transfer fl dev set females have %.f Utts' % pretrainuttsdict["transfer_fl_dev_orgfemale"])
        print('pretrain transfer fl test set males have %.f Utts' % pretrainuttsdict["transfer_fl_test_orgmale"])
        print('pretrain transfer fl test set females have %.f Utts' % pretrainuttsdict["transfer_fl_test_orgfemale"])
        print('pretrain transfer pretrain-train dev set males have %.f Utts' % pretrainuttsdict["transfer_pre_dev_orgmale"])
        print('pretrain transfer pretrain-train dev set females have %.f Utts' % pretrainuttsdict["transfer_pre_dev_orgfemale"])
        print('pretrain transfer pretrain-train test set males have %.f Utts' % pretrainuttsdict["transfer_pre_test_orgmale"])
        print('pretrain transfer pretrain-train test set females have %.f Utts' % pretrainuttsdict["transfer_pre_test_orgfemale"])

        print('FL train set males have %.f Utts' % fluttsdict["trainmale"])
        print('FL train set females have %.f Utts' % fluttsdict["trainfemale"])
        print('FL test set males have %.f Utts' % fluttsdict["testmale"])
        print('FL test set females have %.f Utts' % fluttsdict["testfemale"])
        print('FL dev set males have %.f Utts' % fluttsdict["devmale"])
        print('FL dev set females have %.f Utts' % fluttsdict["devfemale"])

    if stage <= 4:  ## remake whole dataset.
        refjsondir = f"{savedir}/from_train_to_test.json"
        savedir = f"{savedir}/data"
        if not os.path.exists(os.path.join(savedir)):
            os.makedirs(os.path.join(savedir))

        for dset in ["dev", "test"]:
            dsetsavedir = os.path.join(savedir, 'whole', dset)
            if not os.path.exists(os.path.join(dsetsavedir)):
                destination = shutil.copytree(os.path.join(sourcedir, dset + '_org'), os.path.join(dsetsavedir))
        fllist = os.listdir(f"{savedir}/FL")
        traintext = []
        trainwav = []

        for spk in fllist:
            with open(os.path.join(f"{savedir}/FL", spk, 'train', 'text')) as f:
                textlines = f.read().splitlines()
            traintext.extend(textlines)
            with open(os.path.join(f"{savedir}/FL", spk, 'train', 'wav.scp')) as f:
                wavlines = f.read().splitlines()
            trainwav.extend(wavlines)
        with open(os.path.join(f"{savedir}/pretrain", 'train', 'text')) as f:
            textlines = f.read().splitlines()
        traintext.extend(textlines)
        with open(os.path.join(f"{savedir}/pretrain", 'train', 'wav.scp')) as f:
            wavlines = f.read().splitlines()
        trainwav.extend(wavlines)
        traintext.sort()
        trainwav.sort()
        utt2spklist = []
        for i in traintext:
            pspk = i.split(" ")[0]
            utt2spklist.append(pspk + ' ' + pspk.split('-')[0] + '\n')
        savetraindir = os.path.join(savedir, 'whole', 'train')
        if not os.path.exists(os.path.join(savetraindir)):
            os.makedirs(os.path.join(savetraindir))
        utt2spklist.sort()
        traintext.sort()
        trainwav.sort()
        with open(savetraindir + '/utt2spk', 'at') as af:
            af.writelines(utt2spklist)
            af.close()
        with open(savetraindir + '/text', 'at') as af:
            for item in traintext:
                af.write("{}\n".format(item))
            af.close()
        with open(savetraindir + '/wav.scp', 'at') as af:
            for item in trainwav:
                af.write("{}\n".format(item))
            af.close()


























