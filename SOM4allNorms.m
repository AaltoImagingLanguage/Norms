
addpath '~/code/semanticspace/somtoolbox'
addpath '~/code/semanticspace/matfunc'

close all
clear all
%% Common Parameters
msize = 75;
shape = 'toroid';

dpath = '/m/nbe/project/aaltonorms/';
%normset = {'cslb' , 'vinson','aaltoprod', 'aalto85', 
normset = {'cmu', 'w2v_fin', 'w2v_eng'};


for n = 1:length(normset)
vectormat = importdata([dpath 'data/' normset{n} '/vectors.csv']);
vocab  = importdata([dpath 'data/' normset{n} '/vocab.csv']);
[catind, catnames] = getCatInd(vocab, normset{n});

% Compute SOM
sD = som_data_struct(vectormat, 'labels', vocab);
sM = som_make(sD,'munits', msize, 'shape', shape);
%sM = som_make(sD);
[qe(n),te(n)] = som_quality(sM,sD);

som_show_category(catnames,catind,vectormat,vocab, sM , 0, n);
title([normset{n} ' ' num2str(sM.topol.msize) ' ' sM.topol.shape], 'Interpreter', 'none')

outname = [ dpath 'results/SOM/2ndRUN/' normset{n} '_' num2str(sM.topol.msize(1)*sM.topol.msize(2)) sM.topol.shape];
print(outname,'-dpdf','-fillpage')
save(outname, 'sM', 'sD')

end


