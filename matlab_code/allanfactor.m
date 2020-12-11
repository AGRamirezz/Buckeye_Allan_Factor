function [Allan] = allanfactor(fin, numint) 
% Calculates and plots allan factor for event series fin. 
% For scale invariant event series, the allan factor is valid over a wider 
% range of power law exponents than the fanno factor.
% FF is valid for 0< exp <1, while the AF is valid over 0< exp <3 
%   (Teich et al, 1997, J Opt Soc Am A)
% JMB 03/18/02


% get mean spike count
 af = zeros(1, numint);    
 mean = sum(fin)/numint;

% get length of each interval
 int = floor(length(fin)/numint);

% get variance
 tot = 0;
 for i=1:numint-1
     (i-1);
     tot = tot + ( sum(fin((i*int):((i+1)*int)))  -  sum(fin((((i-1)*int) + 1):(i*int))) )^2;
     af(i) = (tot/i)/(2*mean);
 end;
 
% return final Allan factor 
 Allan = af(numint-1);
