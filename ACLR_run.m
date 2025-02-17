clear;clc
%% Select Data, Configure hierarchy h and bipartite graph neighbor k

% Synthetic data
% DataName = 'aggregation'; h = 7;  k = 10;
% DataName = 'dart2';       h = 8;  k = 80;
% DataName = 'boxes';       h = 9;  k = 30;

% Real-world data
DataName = 'control';  h = 8;  k = 36;
% DataName = 'Coil20';   h = 9;  k = 10;
% DataName = 'Yeast';    h = 8;  k = 60;
% DataName = 'MSRA25';   h = 9;  k = 14;
% DataName = 'Waveform'; h = 9;  k = 18;
% DataName = 'PenDigits';h = 10; k = 70;
% DataName = 'MNIST';    h = 11; k = 680;
% DataName = 'Covtype';  h = 10; k = 1190;

%% Load data
load([DataName,'_data.mat']);

%% Conduct ACLR clustering
[result,preY,mY,t,iter] = ACLR(X,label,h,k);

%% Display results
disp([DataName ' results:'])
disp(['ACC = ' num2str(result(1))])
disp(['NMI = ' num2str(result(2))])
disp(['ARI = ' num2str(result(3))])
disp(['iter = ' num2str(iter)])
disp(['t = ' num2str(t)])