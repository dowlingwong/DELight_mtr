import os
import sys

import struct
import numpy as np
from tqdm import tqdm


class FileHeader:
 
    """
        Class including information from the file header
    """
    
    def __init__(self, header):
        self.read_header(header)
        
    def read_header(self, header):
        self.Version = str(header[0])
        self.DataSetRevisionNo = header[3]
        self.Size_FileHeader = header[4]
        self.Size_DataHeader = header[5]
        self.Size_ScopeSettings = header[6]
        self.SamplesPerDataBlock = header[7]
        self.BaseLineAcquisitionRatio = header[8]
        self.BytesPerSample = header[9]
        self.Source = header[10:12]
        self.StartingPointOfAcquisition = header[12]
        self.MaxPixelValue = header[13]
        self.MinPixelValue = header[14]
    
    
class ScopeSettings:
    
    """
        Class including the scope settings
    """

    def __init__(self, header):
        self.read_header(header)
        
    def read_header(self, header):
        self.TriggerHoldoff = header[0]
        self.TriggerSource = header[1:3]
        self.TriggerLevel = header[3:5]
        self.TriggerSlope = header[5:7]
        self.BitsPerSample = header[7:9]
        self.ChannelCoupling = header[9:11]
        self.ChannelImpedance = header[11:13]
        self.Oversampling = header[13]
        self.xfac = header[14]
        self.xofs = header[15]
        self.yfac = header[16:18]
        self.yofs = header[18:20]
        
class DAQReader:
    
    """
    Class loading and reading the raw file from an MMC.
    
    :param file_name: path of the raw file 
    """
    
    def __init__(self, file_name):
        
        self.file_name = file_name
        self.size_file_header = 74
        self.file = open(self.file_name, 'rb')
        self._read_file_header()
        self._read_scope_settings()
        self._get_number_traces()
        
    def _read_file_header(self):
        header = struct.unpack("<22s14s4s4H2I3HI2i", self.file.read(self.size_file_header))
        self.FileHeader = FileHeader(header)
        
    def _read_scope_settings(self):
        header = struct.unpack("<f2H2f9H6d", self.file.read(self.FileHeader.Size_ScopeSettings))
        self.ScopeSettings = ScopeSettings(header)
        
    def _read_data_header(self):
        header = struct.unpack("<6s2IHf2d", self.file.read(self.FileHeader.Size_DataHeader))
        self.Text = str(header[0])
        self.TraceNo = header[1]
        self.TimeStamp = header[2]
        self.Sec100 = header[3]
        self.Temperature = header[4]
        self.xref = header[5]
        self.yref = header[6]
        
    def _get_number_traces(self):
        self.size_trace = self.FileHeader.SamplesPerDataBlock * self.FileHeader.BytesPerSample
        self.number_traces = (os.fstat(self.file.fileno()).st_size - self.size_file_header - self.FileHeader.Size_ScopeSettings)
        self.number_traces /= self.FileHeader.Size_DataHeader + self.size_trace
        self.number_traces = int(self.number_traces)
        
    def read_pulse_i(self, i):
        """
        Read i-th pulse, returning the data array while the data header information is stored in the related 
        attributes (Text, TraceNo, TimeStamp, Sec100, Temperature, xref, yref) of the DAQRead object.
        
        :param i: index of the pulse to read
        """
        data_type = {2: "H", 4: "I"}
        self.file.seek(self.size_file_header + self.FileHeader.Size_ScopeSettings + 
                       i * (self.FileHeader.SamplesPerDataBlock * self.FileHeader.BytesPerSample
                            + self.FileHeader.Size_DataHeader), 0)
        self._read_data_header()
        header = struct.unpack("<" + self.FileHeader.SamplesPerDataBlock * data_type[self.FileHeader.BytesPerSample],
                               self.file.read(self.size_trace))
        return np.array(header)
    
    def read_pulses(self, progress_bar=False, close_file=True):
        """
        Read all the pulses and store them in a structured array "DAQRead.data". 
        After reading all the data, if not differently specified, the file is closed and cannot be accessed anymore.
        
        :param progress_bar: if True, it shows the progress of the data loading (default False)
        :param close_file: if True, it closes the raw file (default True)
        """        
        data_type = {2: "H", 4: "I"}
        dtype = [('Text', '<U6'), ('TraceNo', int), ('TimeStamp', int), ('Sec100', int), ('Temperature', np.double), 
                 ('xref', np.double), ('yref', np.double), ('data', int, (self.FileHeader.SamplesPerDataBlock,))]
        self.data = np.zeros(self.number_traces, dtype=dtype)
        byte_struct = "<" + self.number_traces * ("6s2IHf2d" 
                                            + self.FileHeader.SamplesPerDataBlock * data_type[self.FileHeader.BytesPerSample])
        data_tmp = struct.unpack(byte_struct, 
                                 self.file.read(self.number_traces * (self.FileHeader.Size_DataHeader + self.size_trace)))
        length_data = 7 + self.FileHeader.SamplesPerDataBlock
        gen = tqdm(range(self.number_traces)) if progress_bar else range(self.number_traces)
        for i in gen:
            self.data['Text'][i] = data_tmp[i * length_data]
            self.data['TraceNo'][i] = data_tmp[1 + i * length_data]
            self.data['TimeStamp'][i] = data_tmp[2 + i * length_data]
            self.data['Sec100'][i] = data_tmp[3 + i * length_data]
            self.data['Temperature'][i] = data_tmp[4 + i * length_data]
            self.data['xref'][i] = data_tmp[5 + i * length_data]
            self.data['yref'][i] = data_tmp[6 + i * length_data]
            self.data['data'][i] = data_tmp[7 + i * length_data : 7 + self.FileHeader.SamplesPerDataBlock + i * length_data]
        if close_file:
            self.close()
    
    def close(self):
        self.file.close()