cd C:\Users\Selmaan\Dropbox\Lab\Data\Trained RNN
addpath C:\Users\Selmaan\Documents\GitHub\theano-rnn\trainingRNNs-master

%% Load Net Parameters
clear all,close all,clc,

baseName = 'dCue_(25-75)_';
actFun = @(x) tanh(x);
b_hh = load([baseName 'b_hh.txt'])';
b_hy = load([baseName 'b_hy.txt'])';
W_hh = load([baseName 'W_hh.txt']);
W_hy = load([baseName 'W_hy.txt']);
W_uh = load([baseName 'W_uh.txt']);

%% Simulate Net Trajectories
simTime = 50;
vals = [1;-1];
nUnits = length(b_hh);

for cond = 1:2
    [u , yt] = gendCue(simTime,.33,vals(cond));
    [h,yProb,y] = simRNN(actFun,simTime,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
    sprintf('Target: %0.2f; Model: [%0.2f %0.2f]',u(round(.33*simTime)),yProb(end,1),yProb(end,2)),
end
%% Generate A Trajectory for Each Trial Type
close all,
scaleTrial = 1;
inN = 0;
conditions = 2;

hVec = [];hMaster=[];yMaster=[];
for cond=1:conditions
    [u , yt] = gendCue(simTime,.33,vals(cond));
    u(end+1:simTime*scaleTrial) = 0;
    u = u + randn(size(u,1),1)*inN;
    [h,yProb,y] = simRNN(actFun,scaleTrial*simTime,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
    hMaster(:,:,cond) = h;
    yMaster(:,:,cond) = yProb;
    hVec(end+1:end+simTime*scaleTrial,:) = h;
end

u = randn(simTime*scaleTrial,1)*inN;
[h,yProb,y] = simRNN(actFun,simTime*scaleTrial,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
hMaster(:,:,end+1) = h;
yMaster(:,:,end+1) = yProb;
hVec(end+1:end+simTime*scaleTrial,:) = h;

[e,s,l] = princomp(hVec);
%figure,plot(l/sum(l),'.')
l(1:10)/sum(l),
col = jet(conditions+1);
figure, hold on,
for i=1:conditions+1
    ind = (i-1)*simTime*scaleTrial+1:i*simTime*scaleTrial;
    tOFF = floor(simTime*.33);
    tON = tOFF - floor(simTime*.1);
    plot3(s(ind,1),s(ind,2),s(ind,3),'color',col(i,:),'linewidth',2)
    plot3(s(ind(tOFF),1),s(ind(tOFF),2),s(ind(tOFF),3),'r.','MarkerSize',20)
    plot3(s(ind(tON),1),s(ind(tON),2),s(ind(tON),3),'k.','MarkerSize',20)
end


% %% Analyze Trajectories without typical input
% close all,
% u = randn(simTime*scaleTrial,1)*inN;
% [h,yProb,y] = simRNN(simTime*scaleTrial,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
% figure,imagesc(h),
% figure,imagesc(y),
% eigH = e'*h';
% figure, hold on,
% for i=1:2
%     ind = (i-1)*simTime*scaleTrial+1:i*simTime*scaleTrial;
%     tOFF = floor(simTime*.33);
%     tON = tOFF - floor(simTime*.1);
%     plot3(s(ind,1),s(ind,2),s(ind,3),'color',col(i,:),'linewidth',2)
%     plot3(s(ind(tOFF),1),s(ind(tOFF),2),s(ind(tOFF),3),'r.','MarkerSize',20)
%     plot3(s(ind(tON),1),s(ind(tON),2),s(ind(tON),3),'k.','MarkerSize',20)
% end
% plot3(eigH(:,1),eigH(:,2),eigH(:,3),'k','linewidth',2),


