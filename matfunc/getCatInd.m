function [catind,catnames] = getCatInd(vocab, normset)


fn = '/m/nbe/project/aaltonorms/data/SuperNormList.xlsx'; 
opts = detectImportOptions(fn);       
T = readtable(fn,opts);
catnames = unique(T.category);
catnames(1)= [];
a = 0;
for i = 1:length(catnames)
%    tokens = T(ismember(T.category, catnames{i}),3).eng_name;
    col = find(ismember(opts.SelectedVariableNames, normset));
    fcn = ['T(ismember(T.category, catnames{i}),' num2str(col) ' ).' normset];
    tokens = eval(fcn);
    catind{i} = find(ismember(vocab, tokens));
    a = a+ length(catind{i});

end

if a ~= length(vocab)
   error('some items in the vocab is missing category labels,Check the reference to the look up table')
end
