function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

% AΪ����ֵ�ֽ�ԭʼ����
% cΪѡ��������������
% isMax��������ֵ������С������,isMax = 0�����ǽ������У��������,isMax = 1����
if nargin < 2
    c = size(A,1);
    isMax = 1;       % Ĭ�Ͻ���������
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end

if nargin < 3
    isMax = 1;       % Ĭ�Ͻ���������
    isSym = 1;
end

if nargin < 4
    isSym = 1;
end

if isSym == 1
    A = max(A,A');   % ������ʲô��
end
[v, d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);          % Max = 0��Ӧ����ֵ�������У�ѡ��С��c��
else
    [d1, idx] = sort(d,'descend');% Max = 1��Ӧ����ֵ�������У�ѡ����c��
end

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);