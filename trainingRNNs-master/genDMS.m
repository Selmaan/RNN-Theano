function [u, y_targ] = genDMS(simTime,fracCues,vals)
    
    if nargin<3
        v0=randi(2)*2-3;
        v1=randi(2)*2-3;
    else
        v0 = vals(1);
        v1 = vals(2);
    end

    p0=round(simTime*fracCues(1));
    p1=round(simTime*fracCues(2));
    u = zeros(simTime,1);
    u(p0-round(simTime*.1):p0,1) = v0;
    u(p1:p1+round(simTime*.1),1) = v1;
    %u(p0,1) = v0;
    %u(p1,1) = v1;
    targ = ~(v0 == v1) + 1;
    y_targ(targ) = 1;
end