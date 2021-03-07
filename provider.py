from metrics import *

METRIC_GLOBAL_LIST = [
    "RMSE",
    "Kruskal",
    "Sammon",
    "DTM",
    "DTM_KL1",
    "DTM_KL01",
    "DTM_KL001",
]

METRIC_LOCAL_LIST = [
    "Spearman", 
    "Trustworthiness",
    "Continuity",
    "MRRE_XZ",   ## raw data to embedded data
    "MRRE_ZX",   ## embedded data to raw data
]

class MDPMetricProvider:
    def __init__(self, raw_data, emb_data, metric_list, k=5):
        self.raw = raw_data
        self.emb = emb_data
        self.mlist = metric_list
        self.k = k
        self.result = {}

    
    def run():
        ## Check global / local metric inclusion
        global_metric_set = set(METRIC_GLOBAL_LIST)
        local_metric_set = set(METRIC_LOCAL_LIST)
        global_metric_checklist = []
        local_metric_checklist = []
        for metric in self.mlist:
            if metrics in global_metric_set:
                global_metric_checklist.append(metric)
            elif metrics in local_metric_set:
                local_metric_checklist.append(metric)
            else:
                raise Exception("We currently do not support " + metric + ".")
        
        if len(global_metric_checklist) > 0:
            globalMeasure = GlobalMeasure(self.raw, self.emb)
            for metric in global_metric_checklist:
                score = {
                    "RMSE"      : globalMeasure.rmse(),
                    "Kruskal"   : globalMeasure.kruskal_stress_measure(),
                    "Sammon"    : globalMeasure.sammon_stress(),
                    "DTM"       : globalMeasure.dtm(),
                    "DTM_KL1"   : globalMeasure.dtm_kl(sigma=1.0),
                    "DTM_KL01"  : globalMeasure.dtm_kl(sigma=0.1),
                    "DTM_KL001" : globalMeasure.dtm_kl(sigma=0.01)
                }[metric]
                self.result[metric] = score
        
        if len(local_metric_checklist) > 0:
            localMeasure = LocalMeasure(self.raw, self.emb, self.k)
            for metric in local_metric_checklist:
                score = {
                    "Spearman"        : localMeasure.spearmans_rho(),
                    "Trustworthiness" : localMeasure.trustworthiness(),
                    "Continuity"      : localMeasure.continuity(),
                    "MRRE_XZ"         : localMeasure.mrre_xz(),
                    "MRRE_ZX"         : localMeasure.mrre_zx()
                }[metric]
                self.result[metric] = score

        return self.result

        

        

        


