function [label] = getCatLabel(word)

fn = '/m/nbe/project/aaltonorms/data/SuperNormList.xlsx'; 
opts = detectImportOptions(fn);       
T = readtable(fn,opts);

% look in both fin and eng col
label = T(ismember(T.fin_name, word),4).category{1};
if isempty(label)
label = T(ismember(T.eng_name, word),4).category{1};
end

