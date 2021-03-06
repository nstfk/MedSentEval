'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.mednli import MedNLIEval
from senteval.rqe import RQEEval
from senteval.pubmed20k import PubMed20KEval
from senteval.bioc import BioCEval
from senteval.bioasq import BioASQEval
from senteval.pico import PICOEval
from senteval.clinicalsa import ClinicalSAEval
from senteval.citationsa import CitationSAEval
from senteval.sts import ClinicalSTSEval,BIOSSESEval

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['BioC','CitationSA','ClinicalSA','BioASQ','PICO','PubMed20K','RQE','ClinicalSTS','BIOSSES','MedNLI']
                       

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # MedSentEval tasks
        
        if name == 'MedNLI':
            self.evaluation = MedNLIEval(tpath + '/MedNLI', seed=self.params.seed)
        elif name == 'RQE':
            self.evaluation = RQEEval(tpath + '/RQE', seed=self.params.seed)
        elif name == 'BioC':
            self.evaluation = BioCEval(tpath + '/BIOC', seed=self.params.seed)   
        elif name == 'ClinicalSA':
            self.evaluation = ClinicalSAEval(tpath + '/ClinicalSA', seed=self.params.seed) 
        elif name == 'CitationSA':
            self.evaluation = CitationSAEval(tpath + '/CitationSA', seed=self.params.seed) 
        elif name == 'PubMed20K':
            self.evaluation = PubMed20KEval(tpath + '/PubMed20K', seed=self.params.seed)
        elif name == 'PICO':
            self.evaluation = PICOEval(tpath + '/PICO', seed=self.params.seed)
        elif name == 'BioASQ':
            self.evaluation = BioASQEval(tpath + '/BioASQ', seed=self.params.seed)
        elif name=='ClinicalSTS':
            self.evaluation = ClinicalSTSEval(tpath + '/ClinicalSTS', seed=self.params.seed)
        elif name=='BIOSSES':
            self.evaluation = BIOSSESEval(tpath + '/BIOSSES', seed=self.params.seed)                                   
                           
        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)
        
        print(self.results)
        return self.results
