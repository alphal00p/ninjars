import copy
import unittest
from summarize import validate_spans

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.result={'success':True,'sources_unchanged':True,'union_misses':0,'rounds':10,'width':2,'scope':'shared-bubble-repeated-fixed-point'}
        self.spans={'performance_acceptance':False,'records':[
          {'kind':'phase','label':'cuts::bubble_batch','parent':'none','file':'diagnostic_profile.rs','line':20,'inputs':0,'outputs':0,'calls':10,'rows':20,'inclusive_ns':100,'exclusive_ns':80},
          {'kind':'phase','label':'cuts::tadpole_batch','parent':'none','file':'diagnostic_profile.rs','line':20,'inputs':0,'outputs':0,'calls':10,'rows':40,'inclusive_ns':100,'exclusive_ns':80},
          {'kind':'kernel','label':'exact-evaluator','parent':'cuts::bubble_batch','file':'cuts.rs','line':45,'inputs':8,'outputs':28,'calls':10,'rows':20,'inclusive_ns':20,'exclusive_ns':20}]}
    def test_complete_counter_control(self):
        validate_spans(self.result,self.spans)
    def test_duplicate_missing_and_partial_counters(self):
        for change in ['duplicate','missing','partial']:
            x=copy.deepcopy(self.spans)
            if change=='duplicate':x['records'].append(x['records'][0])
            elif change=='missing':x['records'][0]['rows']=0
            else:x['records'][0]['calls']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):validate_spans(self.result,x)
    def test_collapsed_owner_rejected(self):
        self.spans['records'][-1]['file']='kernel.rs'
        with self.assertRaises(ValueError):validate_spans(self.result,self.spans)
    def test_invalid_self_time_rejected(self):
        self.spans['records'][-1]['exclusive_ns']=21
        with self.assertRaises(ValueError):validate_spans(self.result,self.spans)
    def test_failed_source_guard_rejected(self):
        self.result['sources_unchanged']=False
        with self.assertRaises(ValueError):validate_spans(self.result,self.spans)

if __name__=='__main__':unittest.main()
