function [pp,ax,ay] = peakHilbertPP(w,sr,ws,plen,ppeak,emin,emax)

% emin and emax mean that w will be partitioned into 1/2^emin windows for 
% the longest timescale analyzed, and 1/2^emax windows for the shortest

% plen is the minimum distance between two peaks, in samples. 
% ws is the length of each waveform to be analyzed, in samples

% N waveform segments will be analyzed, where N=ceil(length(w)/ws).
% Segments are left-aligned and tiled except for the last one,
% which is right-aligned.  Mean results across segments are returned.

N = ceil(length(w)/ws);
for i=1:N
    if i==N
        cw = w(end-ws+1:end);
    else
        cw = w((i-1)*ws+1:i*ws);
    end
    
    % compute Hilbert envelope of waveform
    disp(sprintf('compute envelope %d of %d...',i,N));
    %env = abs(hilbert(cw));
    env = cw;
    
    % loop through envelope to find and preserve peaks separated by
    % +/- plen, which serves to skip "blip peaks" in envelope,
    % and also limit the rate of peaks (i.e. amplitude modulations)
    disp('find peaks...');
    n = ws-plen+1;
    for j = 1:n
        [y,idx] = max(env(j:j+plen-1));
        env(j:j+plen-1)=0;
        env(j+idx-1)=y;
    end

    % ppeak is the mean proportion of peaks in each segment.  
    % ppeak is used to set threshold cmax for choosing largest peaks
    envs = sort(env(find(env>0)),'descend');
    cmax = envs(floor(ws*ppeak));
    pp = env>cmax; ppidx = find(pp>0); iei = ppidx(2:end)-ppidx(1:end-1);
    % run Allan Factor variance (aka normalized wavelet variance) 
    %   for each signal and average. 
    %   ax are window sizes T in seconds, assuming sr is in Hertz
    %   ay are AF variances for each T in ax
    disp('run Allan Factors...');
    [ay(:,i),ax(:,i)]=AFanalysis(pp,emin,emax,1/sr);
    
end
if N > 1
    ax=mean(ax'); ay=mean(ay'); 
end
end