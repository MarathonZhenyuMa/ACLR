function [result,preY,mY,t,iter] = ACLR(X,label,h,k,K,Maxiter)
% The code of manuscript "Large-Scale Clustering With Anchor-Based Constrained Laplacian Rank" (ACLR).
% submitted to IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE).
% The code is written by Zhenyu Ma

% Input:
% X: original data num*d
% label: real label num*1
% h: the hierarchy of binary trees for BKHK, so m = 2^h
% k: anchor graph neighbor of B in per column 1<=k<=num
% K: K in K-NN (6 default)
% Maxiter: iteration maximum (30 default)


% Output:
% result: clustering result [ACC NMI ARI]
% preY: predicted labels num*1
% mY: anchor labels 1*m
% t: time overhead
% iter: # iterations

%% Parameter Configuration
[num,~] = size(X);
c = length(unique(label));
if nargin<6
    Maxiter = 30;
end
if nargin<5
    K = 6;
end

tStart = tic;
%% Stage 1: Anchor Points Selection by BKHK
[~,locAnchor] = hKM(X',1:num,h,1);
Z = locAnchor';
m = 2^h;


%% Stage 2: Initial Graph A Construction by TSPT
B = zeros(num,m);
Dis = EuDist2(X,Z,0);
[~,idx] = sort(Dis);
for j = 1:m
    id = idx(1:(k+1),j);    % k+1 smallest distance indexes for i-th sample
    di = Dis(id,j);         % k+1 smallest distances for i-th sample
    B(id,j) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);  % Eq.(35) in Nie et al. 2016 CLR
end

Theta_inv = ones(num,1)./(sum(B,2)+eps);
A = B'*(repmat(Theta_inv,1,m).*B);

lambda = 1;  % lambda initialization
A_sym = (A+A')/2;
L_0 = eye(m)-A_sym;
r0 = rank(L_0);
[H,~,~] = eig1(L_0, c, 0);

S = zeros(m,m);
if r0 < m-c
    error('The original graph has more than %d connected component', c);
end

%% Stage 3: Alternating Optimization for Main ACLR Model
for iter = 1:Maxiter
    % Update S with fixed H
    DisHH = EuDist2(H,H,0);
    DisAH = lambda*DisHH;
    dis = A-DisAH;
    for i = 1:m
        S(i,:) = EProjSimplex_new(dis(i,:));
    end

    % Update H with fixed S
    S_sym = (S+S')/2;
    D_s = diag(sum(S_sym));
    L_s = D_s-S_sym;
    H_old = H;
    [H,~,ev_full] = eig1(L_s, c, 0);

    % Adjust lambda adaptively
    fn1 = ev_full(c);
    fn2 = ev_full(c+1);
    if fn1 > 1e-11          % few large fn1 means connected components < c
        lambda = lambda*2;
    elseif fn2 < 1e-11      % extremely small fn2 means connected components > c
        lambda = lambda/2;H = H_old;
    else                    % =c
        break;
    end
end

% generate anchor labels
G = graph(S_sym);
[mY,~] = conncomp(G);



%% Stage 4: Self-Supervised Label Propagation by K-NN
[~,idxNN] = sort(Dis,2);
idxNNk = idxNN(:,1:K);
NNklabels = zeros(num,K);
for i = 1:num
    for l = 1:K
        NNklabels(i,l) = mY(idxNNk(i,l));
    end
end
preY = mode(NNklabels,2);

t = toc(tStart);
%% Clustering Result
result = ClusteringMeasure_onlyANA(label,preY);