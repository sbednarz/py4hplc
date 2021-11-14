#
# py4hplc
#
# A simple Python 3 library for batch processing of HPLC data with automatic peak detection and integration. 
# This version is designed to process chromatographic results exported to TXT format by ClarityChrom software. 
# Tested for HPLC systems with UV detection (MWD) and RI detection (RID).
# 
# Warning: initial release

import numpy as np
import matplotlib.pyplot as plt
import configparser
import re
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import os
import sys


class Batch:
    
    def __init__(self, configpath):
        
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        
        self.batchdir = os.path.dirname(configpath)
        if not self.batchdir:
           self.batchdir = '.'
        
        if not os.path.exists(configpath):
            raise Exception("Batch config file does not exist: {}".format(configpath))

        self.config.read(configpath)
        if not self.config.sections():
            raise Exception("Batch config error: {}".format(configpath))
        
        self.chromatograms = {}
        
        print("Batch file: {}".format(configpath))
        print("Batch info: {}".format(self.config['header']['info']))
        print("Data directory: {}".format(self.batchdir))
        

        N = len(self.config['chromatograms'].keys())
        i = 1
        for ch_fname in self.config['chromatograms'].keys():
            sampleid = self.config['chromatograms'][ch_fname]
            print("[{}/{}] processing: {} => ".format(i, N, ch_fname), end='')
            ch = Chromatogram(self.batchdir,ch_fname)
            self.__add(ch)
            print("{} ".format(ch.name), end='')
            ch.peaks, ch.bg_a, ch.bg_b, ch.bg_c = self.__process_chromatogram(ch)
            print("")
            i=i+1
            

    def __add(self, chromatogram):
        sampleid = self.config['chromatograms'][chromatogram.filename]
        # sanitize sampleid (Python func name)
        notallowed = "\ +-={}()[].,<>?!\\\/\'\":;|"
        for s in notallowed:
            sampleid = sampleid.replace(s,"_")

        if chromatogram.detector == 'RI':
            channel = 'ri'
        else:
            channel = '' + self.config['uv channels'][chromatogram.channel_no]
            
        if not sampleid in self.chromatograms.keys():
            self.chromatograms[sampleid]={}
            
        name = sampleid+"_"+channel
        if not channel == 'ri':
            chromatogram.channel = float(channel)
        chromatogram.sampleid = sampleid
        chromatogram.name = name
        
        self.chromatograms[sampleid][channel]=chromatogram
        # dynamic attributes
        self.__make_getter(sampleid,channel,name)
        
    def __make_getter(self, sampleid, channel, name):
        def getter(self):
            return self.chromatograms[sampleid][channel]
        setattr(self.__class__, name, property(getter))



    def __process_chromatogram(self, chromatogram):

        dt = chromatogram.dt
        gradient = chromatogram.gradient
        time = chromatogram.time
        signal = chromatogram.signal
        
        # dt*10*60 = 1
        # 10Hz
        # 1 min => 600 parts

        # TODO histogram of signal => autointegrator parameters

        # ----- signal analysis

        histogram1 = np.histogram(gradient, bins=600)
        a,b = histogram1
        j = np.argmax(a)
        #print(j)
        c1 = b[j]
        c2 = b[j+1]
        #print(c1,c2)

        # first pass

        filter = []
        for s in gradient:
            if s > c2:    
                filter.append(False)
            elif s < c1:
                filter.append(False)    
            else:
                filter.append(True)

        t = time[filter]
        s = signal[filter]
        g = gradient[filter]

        #plt.figure(figsize=(9,6))
        #plt.plot(t,s,'.m')

        residues = np.histogram(s,bins=5)

        a,b = residues
        j = np.argmax(a)
        #print(j)
        c1 = b[j]
        c2 = b[j+1]
        #print(c1,c2)


        # second pass
        filter = []
        for y in s:
            if y > c2:    
                filter.append(False)
            elif y < c1:
                filter.append(False)    
            else:
                filter.append(True)


        tnew = t[filter]
        snew = s[filter]
        gnew = g[filter]

        #plt.plot(tnew,snew,'.r')

        # ------ baseline determination

        bg_c, bg_b, bg_a = np.polyfit(tnew, snew, 2)
        tmin = np.amin(tnew)
        tmax = np.amax(tnew)
        t = np.linspace(tmin,tmax)
        
        # baseline second order polynominal
        def bg(x, bg_c, bg_b, bg_a):
            return bg_a + bg_b*x + bg_c*x**2
        
        # the baseline derivative
        def dbg(x, bg_c, bg_b, bg_a):
            return bg_b + 2*bg_c*x
        
        #plt.plot(t, bg_a+bg_b*t+bg_c*t**2,'k--')
        #plt.show()

        #plt.figure(figsize=(9,6))
        #plt.xlim(5,8.5)
        #plt.plot(t, bg_a+bg_b*t+bg_c*t**2,'k.')
        #plt.plot(data[:,0], data[:,1], 'b')
        #plt.show()



        # ------ peak picking
        
        gradient = chromatogram.gradient
        time = chromatogram.time
        signal = chromatogram.signal
        # baseline correction
        #signal = chromatogram.signal - (bg_a + bg_b*time + bg_c*time**2)
        i_max, = np.shape(signal)


        # settings
        
        #GRAD_UP = 2
        #GRAD_DOWN = -2
        #FORWARD = 15
        
        #GRAD_UP = 2
        #GRAD_DOWN = -3
        #FORWARD = 15
       
        # OK!
        #GRAD_UP = 2
        #GRAD_DOWN = -0.3
        #FORWARD = 15

        #Default peak detector settings
        GRAD_UP = 0.7
        GRAD_DOWN = 0.0
        FORWARD = 10
      
        if 'GRAD_UP' in self.config['processing']:
           GRAD_UP = float(self.config['processing']['GRAD_UP'])
        if 'GRAD_DOWN' in self.config['processing']:
           GRAD_DOWN = float(self.config['processing']['GRAD_DOWN'])
        if 'FORWARD' in self.config['processing']:
           FORWARD = int(self.config['processing']['FORWARD'])

        mode = 0
        i = 0
        j = 0

        peak = []
        peaks = {
            "i_start":{},
            "t_start":{},
            "s_start":{},
            "i_end":{},
            "t_end":{},
            "s_end":{},
            "i_max":{},
            "t_max":{},
            "s_max":{},
            "area":{},
            "aa":{},
            "bb":{}
        }

        # dsignal/dt (gradient) analysis
        for g in gradient:

            if mode == 0 and g > dbg(time[i], bg_c, bg_b, bg_a)+GRAD_UP:
                if i+FORWARD < i_max and gradient[i+FORWARD] > dbg(time[i+FORWARD], bg_c, bg_b, bg_a)+GRAD_UP:
                    #print("peak start ", t[i])
                    peak.append(i)            
                    peak.append(time[i])
                    peak.append(signal[i])
                    mode = 1

            if mode == 1 and g <  dbg(time[i], bg_c, bg_b, bg_a)+GRAD_DOWN:
                if i+FORWARD < i_max and gradient[i+FORWARD] > dbg(time[i+FORWARD], bg_c, bg_b, bg_a)+GRAD_DOWN:
                    #print("peak end ", t[i])
                    peak.append(i)            
                    peak.append(time[i])
                    peak.append(signal[i])
                    mode = 0

                    peaks["i_start"][j] = peak[0] 
                    peaks["t_start"][j] = peak[1] 
                    peaks["s_start"][j] = peak[2] 
                    peaks["i_end"][j] = peak[3] 
                    peaks["t_end"][j] = peak[4] 
                    peaks["s_end"][j] = peak[5] 

                    peak = []
                    j = j+1

            i=i+1



        for i in peaks["i_start"].keys():
            i_max = 0
            t_max = 0
            s_max = -1e6
            for j in range(peaks["i_start"][i], peaks["i_end"][i]):
                if signal[j] > s_max:
                    s_max = signal[j-4]
                    t_max = time[j-4]
                    i_max = j-4 # hack

            peaks["i_max"][i] = i_max
            peaks["t_max"][i] = t_max
            peaks["s_max"][i] = s_max


            
        # ------- integration

        from scipy import integrate
        for i in peaks['i_start'].keys():

            t1 =  peaks['t_start'][i]
            t2 =  peaks['t_end'][i]
            s1 =  peaks['s_start'][i]
            s2 =  peaks['s_end'][i]
            aa = (s2-s1)/(t2-t1)
            bb = s1 - ((s2-s1)/(t2-t1))*t1

            ss = signal[peaks['i_start'][i]:peaks['i_end'][i]]
            tt = time[peaks['i_start'][i]:peaks['i_end'][i]]

            signal_bg_corrected = ss - (aa*tt+bb)

            area = trapezoid(signal_bg_corrected, dx=dt)
            peaks['area'][i] = area
            peaks['aa'][i]=aa
            peaks['bb'][i]=bb


        return (peaks, bg_a, bg_b, bg_c)
     
    def peaks (self, samples=[], channels=[], peak_time_range = [], sort_order=[], group_by='sample', outfilename="report.csv", plot=0):
      
       all_chromatograms = self.chromatograms
       
       selected_chromatograms = []
       
       if not samples:
           samples = all_chromatograms.keys()
       
       for s in samples:
           if not channels:
               for channel in all_chromatograms[s]:
                   selected_chromatograms.append(all_chromatograms[s][channel])
           else:
               for channel in channels:
                   if not type(channel) == 'str':
                       channel = str(channel)
                   if not channel in all_chromatograms[s]:
                       raise Exception("Chromatogram of sample {} does not contain channel {}".format(s,channel))
                   selected_chromatograms.append(all_chromatograms[s][channel])
       
       sorted_chromatograms = sorted(selected_chromatograms, key=lambda c: c.name, reverse=False)
       
       
       report = "label,sample_id,filename,start,end,rt,height,area\n"
       
       if group_by == '' or group_by == 'sample':
            
               for c in sorted_chromatograms:
                   sampleid = c.sampleid
                   filename = c.filename
                   label = c.label
                   
                   for p in c.peaks['i_start'].keys():
                       p_start = c.peaks['t_start'][p]
                       p_end = c.peaks['t_end'][p]
                       p_retention_time = c.peaks['t_max'][p]
                       p_height = c.peaks['s_max'][p]
                       p_area = c.peaks['area'][p]
                       
                       if not peak_time_range:
                           peak_time_range = [[-1e6,1e6]] #
                           
                       if peak_time_range:
                           for tr in peak_time_range:
                               if p_retention_time > tr[0] and p_retention_time < tr[1]:
                                   report = report + "\"{}\",\"{}\",\"{}\",{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(label, sampleid, filename, p_start, p_end, p_retention_time, p_height, p_area)
                                   break
                       
                   
                   
                   
       elif group_by == 'channel':
            for c in sorted_chromatograms:
                   print("lol")
       else:
           raise Exception("Unknown group_by option: {}".format(group_by))
       
       f = open(outfilename, "w")
       f.write(report)
       f.close()
        
        
        

    
class Chromatogram:
    
    def __init__(self, batchdir, filename):
        self.file_name = filename
        self.batchdir = batchdir
        self.name = ""
        self.channel = 0
        self.info = {}
        self.peaks = {}
        self.bg_a = 0
        self.bg_b = 0
        self.bg_c = 0
        
        
        
        self.info, tmps, tmpt = self.__read_chromatogram(batchdir, filename)
        t = np.array(tmpt)
        s = np.array(tmps)
        g = np.gradient(s, t[1]-t[0]) # s, dt
        g = savgol_filter(g, 33, 3)
        self.data = np.column_stack((t,s,g))
    
    
    # ClarityChrom TXT file parser
    def __read_chromatogram(self, batchdir, filename):
        path = batchdir+'/'+filename
        with open(path) as f:
            content = f.readlines()
        f.close()

        s = []
        t = []
        info = {}
        
        m = re.match("Sample\ \:\ (.*)", content[2])
        if bool(m):
            info['desc'] = m.groups()[0]
        else:
            raise Exception("File format error: {}".format(filename)) 

        m = re.match("Sample Injected\ \:\ (.*)", content[3])
        if bool(m):
            info['date'] = m.groups()[0]
        else:
            raise Exception("File format error: {}".format(filename))
        
        m = re.match("Inj\. Volume\ \:\ (.*)", content[10])
        if bool(m):
            injvol = m.groups()[0]
            info['injvol'] = float(injvol.replace(",","."))
        else:
            raise Exception("File format error: {}".format(filename))
            
        m = re.match("Data Original\ \:\ (.*)", content[14])
        if bool(m):
            info['fullpath'] = m.groups()[0]
        else:
            raise Exception("File format error: {}".format(filename))

        for line in content[18:]:
            m=re.match("([-]?\d+[\,\.]?\d*)\s([-]?\d+[\,\.]?\d*)", line)
            if bool(m):
                tmpt,tmps = m.groups()
                t.append(float(tmpt.replace(",",".")))
                s.append(float(tmps.replace(",",".")))
            else:
                raise Exception("File format error: {}".format(filename)) 
        return info, s, t
    
    
    @property 
    def dt(self):
        return self.data[1,0] - self.data[0,0]
    @property
    def time(self):
        return self.data[:,0]
    @property
    def t(self):
        return self.data[:,0]
    @property
    def signal(self):
        return self.data[:,1]
    @property
    def s(self):
        return self.data[:,1]
    @property
    def gradient(self):
        return self.data[:,2]
    @property
    def g(self):
        return self.data[:,2]
    @property
    def filename(self):
        return self.file_name
    @property
    def detector(self):
        if bool(re.match(".*RID.*", self.file_name)):
            return 'RI'
        elif bool(re.match(".*MWD.*", self.file_name)):
            return 'UV'
        else:
            raise Exception("Filename error: {}".format(self.file_name))
    @property   
    def channel_no(self):
        match = re.match(".*Channel\ (\d+)", self.file_name)
        if bool(match):
            return match.groups()[0]
        else:
            raise Exception("Filename error: {}".format(self.file_name))
    @property
    def dest(self):
        return self.info['desc']
    @property
    def date(self):
        return self.info['date']
    @property
    def injvol(self):
        return self.info['injvol']
    @property
    def fullpath(self):
        return self.info['fullpath']
    @property
    def label(self):
        if self.detector == 'RI':
            return self.sampleid+" RI"
        else:
            return "{} UV {:.0f} nm".format(self.sampleid, self.channel)
    @property
    def l(self):
        return self.label
    
    @property
    def peaksN(self):
        return len(self.peaks['i_start'].keys())


