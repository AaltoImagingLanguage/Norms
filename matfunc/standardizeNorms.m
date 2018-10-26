function out = standardizeNorms(mat, dim)

if dim == 2
    mat = mat';
end

for i = 1:size(mat,1)
    vari = var(mat(i,:));
    mat(i,:) = mat(i,:)/vari;
    
end

if dim == 2
    mat = mat';
end

out = mat;


end