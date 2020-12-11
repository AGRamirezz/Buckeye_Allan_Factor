function peakHilbertDir(dirfname)
    ds = 4; % downsample by a factor of ds
    minpd = 2.5; % minimal distance between peaks, in ms
    nmin = 4; % number of minutes per AF analysis
    ppeak = 0.005; % proportion of Hilbert envelope peaks extracted
    emin = 4; % 1/2^emin number of largest window partitions
    emax = 14; % 1/2^emax number of smallest window partitions
    nbin = 11; % number of log bins for IEI distribution and 1/f analysis
    fid=fopen(dirfname,'r'); l=fgetl(fid); 
    while ischar(l)
        d1=dir(sprintf('%s/*.mp3',l)); 
        d2=dir(sprintf('%s/*.wav',l)); 
        dd=[d1;d2];
        for i=1:length(dd)
            fname = sprintf('%s/%s',l,dd(i).name);
            disp(sprintf('loading %s...',fname));
            [w, fs] = audioread(fname); sr = fs/ds;
            if nmin>0:
                 dws = sr*60*nmin;
            else
                plen = floor;
            end
            dw = downsample( w(:,1), ds ); 
            if length(dw) < dws
                dws = length(dw);
            end
            [axc,ayc,pxc,pyc,sxc,syc] = peakHilbertPP(dw,sr,dws,plen,ppeak,emin,emax,nbin);
            if size(w,2) > 1
                dw = downsample( w(:,2), ds ); 
                [ax2,ay2,px2,py2,sx2,sy2] = peakHilbertPP(dw,sr,dws,plen,ppeak,emin,emax,nbin);
                axc=(axc+ax2)/2; ayc=(ayc+ay2)/2; 
                pxc=(pxc+px2)/2; pyc=(pyc+py2)/2;
                sxc=(sxc+sx2)/2; syc=(syc+sy2)/2;
            end
            ax(i,:)=axc; ay(i,:)=ayc; 
            px(i,:)=pxc; py(i,:)=pyc;
            sx(i,:)=sxc; sy(i,:)=syc;
            apf(i,:)=polyfit(log(axc),log(ayc),3);
            apv(i,:)=polyval(apf(i,:),log(axc));
            ppf(i,:)=polyfit(log(pxc),log(pyc),3);
            ppv(i,:)=polyval(ppf(i,:),log(pxc));
            spf(i,:)=polyfit(log(sxc),log(syc),3);
            spv(i,:)=polyval(spf(i,:),log(sxc));
        end
        clear d1 d2 dw w axc ayc pxc pyc sxc syc ax2 ay2 px2 py2 sx2 sy2;
        save(sprintf('%s/afdir.mat',l)); 
        clear ax ay px py apf apv ppf ppv spf spv;
        l=fgetl(fid);
    end
    fclose(fid);
end
