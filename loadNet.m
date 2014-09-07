%% Load Net Parameters
clear all,close all,clc,

baseName = 'dCue001_';
actFun = @(x) tanh(x);
b_h = load([baseName 'b_h.txt'])';
b_y = load([baseName 'b_y.txt'])';
W_hh = load([baseName 'W_hh.txt']);
W_out = load([baseName 'W_out.txt']);
W_in = load([baseName 'W_in.txt']);
h0 = load([baseName 'h0.txt']);
cueOn = 0.3;
cueOff = 0.4;
decideOn = 0.9;

%%
simTime = 100;
uIn = zeros(simTime,2,2);
uIn(round(cueOn*simTime):round(cueOff*simTime),1,1) = 1;
uIn(round(cueOn*simTime):round(cueOff*simTime),2,2) = 1;

h(1,:) = h0;
for t = 2:simTime
    h(t,:) = actFun(h(t-1,:)*W_hh + uIn(t,:,2)*W_in + b_h);
end
y = h*W_out + repmat(b_y,simTime,1);

figure,imagesc(h(2:end,:)),
figure,plot(y(2:end,:))