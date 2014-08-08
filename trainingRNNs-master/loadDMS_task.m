cd C:\Users\Selmaan\Dropbox\Lab\Data\Trained RNN
addpath C:\Users\Selmaan\Documents\GitHub\theano-rnn\trainingRNNs-master

%% Load Net Parameters
clear all,close all,clc,

baseName = 'dCue_(50)_';
b_hh = load([baseName 'b_hh.txt'])';
b_hy = load([baseName 'b_hy.txt'])';
W_hh = load([baseName 'W_hh.txt']);
W_hy = load([baseName 'W_hy.txt']);
W_uh = load([baseName 'W_uh.txt']);

%% Simulate Net Trajectories
simTime = 50;
vals = [1 1;1 -1;-1 1;-1 -1];
nUnits = length(b_hh);

for cond = 1:4
    [u , yt] = genDMS(simTime,[.33 .66],vals(cond,:));
    [h,yProb,y] = simRNN(simTime,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
    sprintf('Target: [%0.2f %0.2f]; Model: [%0.2f %0.2f]',u(round(.33*simTime)),u(round(.66*simTime)),yProb(end,1),yProb(end,2)),
end
%% Generate A Trajectory for Each Trial Type
hVec = [];hMaster=[];yMaster=[];
for cond=1:4
    [u , yt] = genDMS(simTime,[.33 .66],vals(cond,:));
    [h,yProb,y] = simRNN(simTime,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy);
    hMaster(:,:,cond) = h;
    yMaster(:,:,cond) = yProb;
    hVec(end+1:end+simTime,:) = h;
end

[e,s,l] = princomp(hVec);
figure,plot(l/sum(l),'.')
col = jet(4);
figure, hold on,
for i=1:4
    ind = (i-1)*simTime+1:i*simTime;
    plot3(s(ind,1),s(ind,2),s(ind,3),'color',col(i,:),'linewidth',2)
end
    