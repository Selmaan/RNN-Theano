function [u, y_targ] = gendCue(simTime,fracCues,vals)
    
    if nargin<3
        v0=randi(2)*2-3;
    else
        v0 = vals(1);
    end

    p0=round(simTime*fracCues(1));
    u = zeros(simTime,1);
    u(p0-round(simTime*.1):p0,1) = v0;
    %u(p0,1) = v0 * 5;
    targ = v0/2 + 1.5;
    y_targ(targ) = 1;
end