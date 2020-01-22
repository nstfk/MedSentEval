# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
MedNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier


class MedNLIEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : MedNLI Entailment*****\n\n')
        self.seed = seed
        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))

        trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                              encoding='utf-8').read().splitlines()
        trainids = io.open(os.path.join(taskpath, 'ids.train'),
                              encoding='utf-8').read().splitlines()
        

        valid1 = self.loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = self.loadFile(os.path.join(taskpath, 's2.dev'))
        validlabels = io.open(os.path.join(taskpath, 'labels.dev'),
                              encoding='utf-8').read().splitlines()
        
        validids = io.open(os.path.join(taskpath, 'ids.dev'),
                              encoding='utf-8').read().splitlines()

        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))
        testlabels = io.open(os.path.join(taskpath, 'labels.test'),
                             encoding='utf-8').read().splitlines()
        testids = io.open(os.path.join(taskpath, 'ids.test'),
                              encoding='utf-8').read().splitlines()
        
        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels,trainids),
                              key=lambda z: (len(z[0]), len(z[1]), z[2],z[3]))
        train2, train1, trainlabels, trainids = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels,validids),
                              key=lambda z: (len(z[0]), len(z[1]), z[2],z[3]))
        valid2, valid1, validlabels, validids = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels,testids),
                             key=lambda z: (len(z[0]), len(z[1]), z[2],z[3]))
        test2, test1, testlabels,testids = map(list, zip(*sorted_test))
        #print(testids)

        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {'train': (train1, train2, trainlabels,trainids),
                     'valid': (valid1, valid2, validlabels,validids),
                     'test': (test1, test2, testlabels,testids)
                     }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.split() for line in
                    f.read().splitlines()]

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels,ids = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0,n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (200*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            try:
                self.y[key] = [dico_label[y] for y in mylabels]
            except:
                logging.info(' key error')
                continue


        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        
        devacc, testacc,yhat,probs= clf.run()
        
        pred=[]
        print(self.data['test'][0])
        print(self.data['test'][1])
        print(yhat)
        for i in yhat:
            pred.append(i)
        print(pred)
        print(probs)
       
        logging.debug('Dev acc : {0} Test acc : {1} for MedNLI\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
