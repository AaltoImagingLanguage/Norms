function [tokens] = getCatMembers(catname)

%returns all members in a particular category

fn = '/m/nbe/project/aaltonorms/data/SuperNormList.xlsx'; 
opts = detectImportOptions(fn);       
T = readtable(fn,opts);
tokens = T(ismember(T.category, catname),:);
