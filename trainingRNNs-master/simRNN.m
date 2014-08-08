function [h,yProb,y] = simRNN(simTime,nUnits,u,W_uh,W_hh,W_hy,b_hh,b_hy)

h = zeros(simTime,nUnits);
h(1,:) = tanh(u(1,:)*W_uh + b_hh);
for i=2:simTime
    h(i,:) = tanh(h(i-1,:)*W_hh + u(i,:)*W_uh + b_hh);
end
y = h*W_hy + repmat(b_hy,simTime,1);
yProb = exp(y)./repmat(sum(exp(y),2),1,2);