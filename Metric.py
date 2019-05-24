__author__ = 'Lin'


import re



def getEvalData(sen):
    b = re.findall(r'\d+',sen)
    cur_new = []
    x = 0
    while x < len(b):
        cur_new.append(b[x]+'-'+b[x+1])
        x = x+2 
    span = re.split(r' ',sen)
#    print(span)
    dic = {}
    for i in range(len(span)):
        temp = span[i]
        IDK = re.split(r'[:,=]',temp)
        Nuclearity1 = IDK[1]
        relation1 = IDK[2]
        Nuclearity2 = IDK[5]
        relation2 = IDK[6]
        dic[cur_new[2*i]]=[relation1,Nuclearity1]
        dic[cur_new[2*i+1]]=[relation2,Nuclearity2]
    return dic



def getMeasurement(sen1,sen2):
    
	dic1 = getEvalData(sen1)
	dic2 = getEvalData(sen2)
	NoNS = 0
	NoRelation = 0
    
	# no of right spans
	RightSpan = list(set(dic1.keys()).intersection(set(dic2.keys())))
	NoSpans = len(RightSpan)


	# Right Number of relaitons and nulearity
	for span in RightSpan:
		if dic1[span][0] == dic2[span][0]:
			NoRelation = NoRelation + 1
		if dic1[span][1] == dic2[span][1]:
			NoNS = NoNS + 1


	# Measurement
	correct_span = NoSpans
	correct_relation = NoRelation
	correct_nuclearity = NoNS
	no_system = len(dic1.keys())  
	no_golden = len(dic2.keys())
    
    # return numbers
	return correct_span, correct_relation, correct_nuclearity, no_system, no_golden
     


def getBatchMeasure(Spans_batch, GoldenMetric_batch):
    
    correct_span =  0
    correct_relation = 0
    correct_nuclearity= 0
    no_system = 0
    no_golden = 0
    
   
    
    for i in range(len(Spans_batch)):
        
        cur_sent = Spans_batch[i][0]
        cur_golden = GoldenMetric_batch[i][0]
        
        if cur_sent != 'NONE' and cur_golden != 'NONE':
        
            cur_spanno, cur_relationno, cur_NSno, cur_sysno, cur_goldenno = getMeasurement(cur_sent, cur_golden)
            
            correct_span =  correct_span + cur_spanno
            correct_relation = correct_relation + cur_relationno
            correct_nuclearity = correct_nuclearity + cur_NSno
            no_system = no_system + cur_sysno
            no_golden = no_golden + cur_goldenno
            
        elif cur_sent != 'NONE' and cur_golden == 'NONE':
            
            _, _, _, cur_sysno, _ = getMeasurement(cur_sent, cur_sent)
            no_system = no_system + cur_sysno
            
        elif cur_sent == 'NONE' and cur_golden != 'NONE':
            
            _, _, _, cur_goldenno = getMeasurement(cur_golden, cur_golden)
            no_golden = no_golden + cur_goldenno
            

    return correct_span, correct_relation, correct_nuclearity, no_system, no_golden
    

def getMicroMeasure(correct_span,correct_relation,correct_nuclearity,no_system,no_golden):
    # Computer Micro-average measure    
    # Span
    Precision_span = correct_span / no_system
    Recall_span = correct_span / no_golden
    F1_span = (2 * correct_span) / (no_golden + no_system)

	# Relation
    Precision_relation = correct_relation / no_system
    Recall_relation = correct_relation / no_golden
    F1_relation = (2 * correct_relation) / (no_golden + no_system)

	# Nuclearity
    Precision_nuclearity = correct_nuclearity / no_system
    Recall_nuclearity = correct_nuclearity / no_golden
    F1_nuclearity = (2 * correct_nuclearity) / (no_golden + no_system)
        
        
    return (Precision_span,Recall_span,F1_span),(Precision_relation,Recall_relation,F1_relation),\
            (Precision_nuclearity,Recall_nuclearity,F1_nuclearity)
        
        
