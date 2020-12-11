function [actual, abcis] = AFanalysis(fin,start,powers,binwidth)
% calculates allan factor over a range of window sizes, given by powers.
% takes a binary input train, fin. 
% actual are the allan factors from actual data
% abcis are the sizes of the counting windows (the x-axis is the abcissa) 
% JMB 03/19/02

% start is the minimum number of divisions of the waveform,
% e.g. if start = 4, then there are 2^4 = 16 divisions,
% which is the recommended minimum

% powers and base set the number of divisions in the recording time. 
% base^powers = max number of divisions
base = 2;   
            
Allan1 = zeros(1, length(start:powers));
count = 1;
for i=start:powers
    abcissa(count) = base^(i);
    Allan1(count) = allanfactor(fin, base^(i));
    count = count + 1;
end;

len = length(abcissa);
for i=1:len
    actual(i) = Allan1(len + 1 - i);
    abcis(i) = (length(fin)*binwidth)/abcissa(len + 1 - i);
end;
