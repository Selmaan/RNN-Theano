%%
close all,
clear all,
clc

nNeurons = 1e3;
W = randn(nNeurons)/sqrt(nNeurons);
[V,D] = eig(W);
D = diag(D);
[~,maxInd] = max(real(D));
D(maxInd),
D(maxInd+1),
%%
close all,
thresh = 0;
actFun= @(x) x.*(x>thresh);
actFun = @(x) (x.*(x>thresh));
tau = 10;
simTime = 2e3;
W_sim = 2*W;
%%
x=nan(nNeurons,simTime);
x(:,1) = randn(nNeurons,1)/1;
b = base*0;
%b = b(randperm(nNeurons));
for t = 1:simTime-1
    dx = -x(:,t) + W_sim*actFun(x(:,t)) + b;
    x(:,t+1) = x(:,t) + dx/tau;
end

figure,imagesc(x(:,tau*5:end)),
[e,s,l] = princomp(x(:,tau*5:end)');
l(1:10)/sum(l),
figure,scatter(s(:,1),s(:,2),15,1:size(s,1),'filled')