function som_show_category(catnames,catind, vectormat,vocab, sM , hit,n )
%
% Plots the SOM hit or labels by predefined category color
% author: Annika Hultén
%
% Input arguments: 
% catnames:  cellarray with category names (string)
% catind:    cellarray with the length of #categories, where each cell contains
%            the vocab indeces of that category
% vectormat: feature data in the form words x features
% vocab:     cell array with vocabulary
% sM:        precomputed SOM struct
% hit:       logical marker of whether to plot hit (1) or lables(0)
% n:        figure number, optional, default 1
%
if nargin < 6
    error('The function requires minimally input parameters')
end

if nargin < 7
n = 1;
end
   
figure(n);
fig = gcf;
set(fig, 'Units','centimeters','Position',[80 10 30 33]);
clr = flip(colormap(fig, hsv(length(catnames))));

aa = get(gca, 'Parent');
set(aa, 'color',[0.7 0.7 0.7]);

som_show(sM, 'empty', '');
y = 0;

for i= 1:length(catnames)
    
    bmus2 = som_bmus(sM,vectormat(catind{i},:), 'best');
    hits2 = som_hits(sM,vectormat(catind{i},:), 'crisp');
    sM2 = som_label(sM,'add', bmus2, vocab(catind{i}));
    
    if hit % plot hit diagram
        som_show_add2('hit', hits2,'MarkerColor',  clr(i,:));    
    else % plot scaled text
        % Compute the distance beween each word and all cell centers
        D = pdist2(vectormat, sM2.codebook, 'euclidean');
        som_show_add_marijn('label', sM2, 'distances', D, 'vocab',vocab,...
            'textsize', 4,'TextColor',  clr(i,:) );
    end
    
%     hold on
%     bb = get(gca,'Children');
%     set(bb, 'EdgeColor', 'white');
    
    str = sprintf('%s', char(catnames(i)));
    y = y + 0.4;
    text(-0.6, y , str, 'Color', clr(i,:) )
    hold on 
    
end





