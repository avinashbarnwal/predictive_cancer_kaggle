import math

def entropy(Var):
    var_unique = Var.unique()
    count_var = dict()
    n = len(var_unique)
    total = len(Var)
    for i in var_unique:
        count_var[i]= sum(1 for v in Var if v==i) 
    entropy = 0
    for i in var_unique:
        p = float(count_var[i])/total
        #print(p)
        entropy = entropy + p*math.log(p)
    entropy = -1*entropy    
    return entropy

def Information_Gain(Var,Class):
    bin_var =  pd.qcut(Var, 10, labels=False,duplicates='drop')
    bin_unique = bin_var.unique()
    ids = dict()
    entropy_var = dict()
    count = dict()
    total=len(Var)
    for i in bin_unique:
        ids[i] = (np.where(i==bin_var))[0]
        count[i] = len(ids[i])
        #print(ids[i])
        #print(Class[ids[i]])
        entropy_var[i] = entropy(Class[ids[i]])
    
    entropy_sup = entropy(Class)
    entropy_cal = 0
    
    for i in bin_unique:
        p = float(count[i])/total
        entropy_cal = entropy_cal + p*entropy_var[i]
        
    
    iv = entropy_sup - entropy_cal
    
    return iv

entropy_varlist =  dict()
columns  =  list(selected_feature_space.columns.values)
print(columns)
for i in columns:
    entropy_varlist[i] = Information_Gain (selected_feature_space[i],data_text['Class'])
        
print(entropy_varlist)

entropy_final = pd.DataFrame({'features':entropy_varlist.keys(),'values':entropy_varlist.values()})
print(entropy_final)

entropy_final.to_csv("entropy_final.csv",sep=",")
print(selected_features)
