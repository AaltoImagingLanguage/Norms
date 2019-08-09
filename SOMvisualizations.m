addpath '/m/nbe/project/corpora/matlab/somtoolbox'
addpath '/m/nbe/project/corpora/matlab'
addpath 'matfunc/'

close all
clear all
%% Common Parameters
msize = 75;
shape = 'toroid';

dpath = '/m/nbe/project/aaltonorms/';
normset = {'aaltoprod'}


for n = 1:length(normset)
    vectormat = importdata([dpath 'data/' normset{n} '/vectors.csv']);
    vocab  = importdata([dpath 'data/' normset{n} '/vocab.csv']);
    [catind, catnames] = getCatInd(vocab, normset{n});
    data = textscan(fopen([dpath 'data/' normset{n} '/correspondence.csv'], 'r'), ...
    '%n%s%s%*[^\n]', 'delimiter', '\t', 'headerlines', 1)
    envocab = data{3}; 
end


mycolormap = [230, 25, 75; ...
              60, 180, 75; ...
              255, 225, 25; ... 
              0, 130, 200; ...
              245, 130, 48; ...
              145, 30, 180; ...
              70, 240, 240; ...
              240, 50, 230; ...
              210, 245, 60; ...
              250, 190, 190; ...
              0, 128, 128; ...
              230, 190, 255; ...
              255, 250, 200; ...
              128, 0, 0; ...
              170, 255, 195; ...
              255, 215, 180; ...
              0, 0, 128]./255;  


%abstract/concrete dichotomy
concrete = catind(find(cellfun(@isempty, strfind(catnames, ...
                                                 'abstract'))));
concrete = cat(1,concrete{:}); 

abstract = catind((find(~cellfun(@isempty, strfind(catnames, ...
                                                  'abstract')))));
abstract = cat(1,abstract{:}); 

catgids = {abstract, concrete}

%sD = som_data_struct(vectormat, 'labels', vocab);
%sD = som_normalize(sD, 'var'); 
%sM = som_make(sD,'munits',msize, 'shape', shape);

%[qe(n),te(n)] = som_quality(sM,sD);

%sM2 = som_autolabel(sM, sD); 

%figure; 
%som_show(sM, 'empty', '', 'footnote', normset{n}); 
%som_show_add('label', sM2.labels)
 



%figure; 
%baa = som_show(sM,'empty','','empty','', 'footnote','aaltonorms', 'edge', 'off'); 

%catgnames = {'abstract', 'concrete'};
%for i = 1:2
  
%    bmus2 = som_bmus(sM,vectormat(catgids{i},:), 'best'); 
%    hits2 = som_hits(sM,vectormat(catgids{i},:), 'crisp');
%    sM2 = som_label(sM,'add', bmus2, vocab(catgids{i})); 

    %kommentoi tämä jos haluat ilman hit-kuvaajia
    %   som_show_add2('hit', hits2, 'Subplot', i, 'MarkerColor', 'black'); 
    % bb = get(gca,'Children');
    %set(bb(2), 'EdgeColor', [0.7, 0.7, 0.7]); 

    %str = sprintf('%s', char(catgnames(i))); 
    %title(str);
    %end


%pieplot

%for i = 1:2
%hits3(i,:) = som_hits(sM2,vectormat(catgids{i},:), 'crisp');
%end
%figure; 
%som_show(sM2, 'empty', '', 'footnote',  normset{n}); 
%som_show_add('hit', hits3', 'marker', 'pie', 'MarkerColor', [0,0,0;0.7, ...
%                  0.7, 0.7])

%%Train map only on Abstract words

catg = 1 % abstract
Xn = vectormat(catgids{catg},:);

sData = som_data_struct(Xn, 'labels', envocab(catgids{catg}));
%sD = som_normalize(sD, 'var'); 
sMap = som_make(sData,'munits',60, 'shape', shape);

[qe(n),te(n)] = som_quality(sMap,sData);

sMap2 = som_autolabel(sMap, sData); 
colormap(bone)
figure; 
som_show(sMap, 'empty', '', 'footnote', strcat(normset{n}, '-abstract')); 
som_show_add('label', sMap2.labels)

%high vs mid-abstract - high are the first 50 words in the abstract category

abstractness={1:50;51:99}; 
abstractlabels={'highly abstract', 'medium abstract'};


for i = 1:2
hits2(i,:) = som_hits(sMap2,Xn(abstractness{i},:), 'crisp');
end

[c,p,err,ind] = kmeans_clusters(sMap, 10)
[dummy, id] = min(ind)
mycolors=colormap(bone(id)); 
abstract_colormap=[mycolormap(12,:); mycolormap(15,:)]; 
som_show(sMap,'color',{p{id},sprintf('%d clusters',id)}, 'footnote', ...
         '', 'colormap', mycolors)

som_show_add('hit', hits2', 'marker', 'pie', 'MarkerColor', ...
             abstract_colormap)
som_show_add('label', sMap2.labels, 'textcolor', 'black')


for ii = 1:length(abstractlabels)
    mypatch1(ii) = patch(NaN, NaN,abstract_colormap(ii,:));
end
legend(mypatch1, abstractlabels, 'location', 'northeastoutside');

figure; 
som_show(sMap2, 'empty', '', 'footnote',  strcat('abstract-',normset{n})); 
som_show_add('hit', hits2', 'marker', 'pie', 'MarkerColor', [0,0,0;0.7, ...
                   0.7, 0.7])

som_show_add('label', sMap2.labels, 'textcolor', 'blue')


%% Just concrete 

aaltoprod_catids=find(not(cellfun(@isempty, catind)))
aaltoprod_concreteids=aaltoprod_catids(find(cellfun(@isempty, ...
                                                strfind(catnames(aaltoprod_catids), 'abstr'))))
aaltoprod_concretelabels = catnames(aaltoprod_concreteids)

catg = 2 % concrete
Xn = vectormat(catgids{catg},:);

sData = som_data_struct(Xn, 'labels', envocab(catgids{catg}));
%sD = som_normalize(sD, 'var'); 
sMap = som_make(sData,'munits',60, 'shape', shape);

[qe(n),te(n)] = som_quality(sMap,sData);

sMap2 = som_autolabel(sMap, sData); 
colormap(spring)
figure; 
som_show(sMap, 'empty', '', 'footnote', strcat(normset{n}, '-concrete')); 
som_show_add('label', sMap2.labels)

%concrete categories


for i = 1:length(aaltoprod_concretelabels)
    categlen(i) = length(catind{find(strcmp(catnames, ...
                                            aaltoprod_concretelabels{i}))}); 
end

for i = 1:length(categlen)
    
    if i == 1; 
        concretecatg(i) = {1: categlen(1)};
    else
        concretecatg(i) = { sum(categlen(1:i-1))+1: ...
                           sum(categlen(1:i))};
    end
end
%mycolormap = colormap(colorcube(17)) 


for i = 1:length(concretecatg)
hits(i,:) = som_hits(sMap2,Xn(concretecatg{i},:), 'crisp');
end
colormap(bone)
[c,p,err,ind] = kmeans_clusters(sMap, 10)
[dummy, id] = min(ind)
mycolors = colormap(bone(id)); 
som_show(sMap,'color',{p{id},sprintf('%d clusters',id)}, 'footnote','', 'colormap',mycolors)
som_show_add('hit', hits', 'marker', 'pie', 'markercolor', mycolormap)
%som_show_add('label', sMap2.labels, 'textcolor', 'blue')


for ii = 1:size(mycolormap,1)
    mypatch(ii) = patch(NaN, NaN, mycolormap(ii,:));
end
legend(mypatch, aaltoprod_concretelabels, 'location', 'northeastoutside');

%cb = colorbar;
%set(cb, 'ticks', 1:id, 'ticklabels', []));
%set(gca, 'clim', [0.5 5.5]);

save(strcat(dpath,'figures/'datestr(now,'dd.mm.yyyy', 2), ...
            '__abstr_concrete_aaltonorms.mat')); 

