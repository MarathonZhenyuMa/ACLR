function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

% A为特征值分解原始矩阵；
% c为选择特征向量数；
% isMax决定特征值升序（最小化问题,isMax = 0）还是降序排列（最大化问题,isMax = 1）；
if nargin < 2
    c = size(A,1);
    isMax = 1;       % 默认解决最大化问题
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end

if nargin < 3
    isMax = 1;       % 默认解决最大化问题
    isSym = 1;
end

if nargin < 4
    isSym = 1;
end

if isSym == 1
    A = max(A,A');   % 作用是什么？
end
[v, d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);          % Max = 0对应特征值升序排列，选最小的c个
else
    [d1, idx] = sort(d,'descend');% Max = 1对应特征值降序排列，选最大的c个
end

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);