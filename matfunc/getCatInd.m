function [catind,catnames] = getCatInd(vocab, normset)
%this function requires matlan 2017a or newer to work

fn = '/m/nbe/project/aaltonorms/data/SuperNormList.xlsx'; 
opts = detectImportOptions(fn);       
T = readtable(fn,opts);
catnames = unique(T.category);
%catnames(1) = [];

a = 0;
col = find(ismember(opts.SelectedVariableNames, normset));


for i = 1:length(catnames)
    
    fcn = ['T(ismember(T.category, catnames{i}),' num2str(col) ' ).' normset];
    tokens = eval(fcn);
    tokens(ismember(tokens, 'NA'))= [];
    catind{i} = find(ismember(vocab, tokens));
    a = a+ length(catind{i});
end

if a ~= length(vocab)
   error('some items in the vocab is missing category labels. Check the reference to the look up table')
end

% % % normset = 'w2v_eng'
% % fin = T.w2v_eng;
% fin = T.cmu;
% fin(ismember(fin, 'NA'))= [];
% df = find(~ismember(fin, vocab));
% 
% [uniqueList,~,uniqueNdx] = unique(fin);
% N = histc(uniqueNdx,1:numel(uniqueList));
% dupNames = uniqueList(N>1);