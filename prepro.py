import json,os,time,pickle,time
import numpy as np
import scipy.io.wavfile as wavfile

def compute_log_mel_fbank_fromsig(signal, sample_rate,n=80):
    # 2.预增强
    # signal = signal[:512]
    # print(signal)

    MEL_N = n

    # 3.分帧
    frame_size, frame_stride = 0.032, 0.008
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,1)
    frames = pad_signal[indices]


    pre_emphasis = 0.97
    for i in range(frames.shape[0]):
        frames[i] = np.append(frames[i][0], frames[i][1:] - pre_emphasis * frames[i][:-1])

    # 4.加窗
    hamming = np.hamming(frame_length)
    frames *= hamming

    # 5.N点快速傅里叶变换（N-FFT）
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((mag_frames ** 2))  # 获取能量谱  (1.0 / NFFT) *
    # 6.提取mel Fbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    n_filter = MEL_N  # mel滤波器组的个数, 影响每一帧输出维度，通常取40或80个
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((n_filter, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, n_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    # 7.提取log mel Fbank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 1 * np.log(filter_banks) -3  # dB
    filter_banks[np.where(filter_banks<0)]=0
    filter_banks=filter_banks*10.0
    # print(filter_banks.astype(np.uint8))
    filter_banks = filter_banks.astype(np.uint8)

    # filter_banks = (filter_banks - 127.5) * 0.0078125   #归一化

    filter_banks = filter_banks *0.0039215686  # 归一化
    # print(filter_banks.dtype)
    return filter_banks

def compute_log_mel_fbank(wav_file,n=80,noisy = 0):
    """
    计算音频文件的fbank特征
    :param wav_file: 音频文件
    :return:
    """
    # 1.数据读取
    sample_rate, signal = wavfile.read(wav_file)
    if noisy == 1:
        fbank = compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)
        out = []
        out.append(fbank)
        for i in range(9):
            signal = add_noise(signal, w=0.001*(i+1))
            fbank = compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)
            out.append(fbank)

        return np.array(out)
    else:
        return compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)

def add_noise(x, w=0.008):
    # w：噪声因子
    x = x/32768
    output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
    return output*32768


from PIL import Image
# a = compute_log_mel_fbank('./wav/bx10.wav',80,0)
# print(a[255])
# a = a.reshape(-1,80,1)
# a = np.concatenate((a,a,a),2)
# print(a.flatten())
# b=Image.fromarray(a[:500])

# b = b.convert('RGB')
# b.save('./bx0.bmp')
#
#
# c = Image.open('images/pic030.bmp')
#
# c = np.array(c)
# print(c.shape)
# print(c[:,:,:1])
# print((c - 127.5) * 0.0078125)


# signal = np.array([2821,-1376,-2921,3465,-313,-2261,1373,728,754,-1139,-2599,3388,2429,-3372,216,585,-1234,614,-26,479,646,-2940,-384,4371,-108,-2671,-11,1282,806,-581,1439,2558,-4515,-1393,3494,-1241,2371,1318,-3705,918,641,-2897,2332,2923,-1436,-1490,494,390,-1695,733,1117,-2854,1107,2172,-2935,1253,1584,-4814,-940,5707,-985,-1630,4233,-3502,-3854,3789,591,-223,1967,-2703,-176,3851,-1646,-1659,2094,-605,-1923,2736,3094,-2945,-3485,853,-764,-683,4358,-1190,-3265,214,-396,655,1479,-375,-1837,-805,552,1609,-903,-1320,-1175,-1929,1598,2183,-1262,-1550,-1220,-222,1455,-167,2135,-1241,-4191,1436,1084,-692,1878,-34,-3880,-298,2659,-1315,-1788,4214,-1053,-4999,2549,2668,-3857,-1444,3646,-3514,-150,2477,-3754,387,1386,-3744,397,3543,-2904,-1384,2801,-1240,-2726,2233,1095,-2689,2046,1276,-4490,1517,2526,-4203,1220,3508,-2973,-1432,2658,-681,-3007,1310,2747,-1259,-1696,313,-1002,-621,1734,-340,-1103,907,-2176,-674,2959,-532,-2521,407,1381,-296,587,-241,-1393,-1238,779,1062,304,-298,-1280,-655,44,556,812,480,-1358,-934,490,655,-412,-1419,-21,621,673,-1192,42,599,-2366,-306,2766,1254,-2133,-1755,345,887,851,1585,241,-1682,-810,755,1606,-38,-648,-686,573,1100,587,-524,-1293,-738,185,2054,2283,-595,-2579,69,1375,548,424,431,26,-1016,371,3182,1236,-2617,-851,1875,1635,1366,648,-1436,-1057,272,1059,2251,627,-1387,-1357,110,1664,1592,-88,-414,530,-24,557,1259,-348,-1113,337,733,149,-73,-37,-779,-945,902,1237,-230,-846,330,1239,-12,-8,325,-342,599,638,188,420,-13,-626,-136,1040,516,-555,-833,315,585,-340,-196,520,-5,-1109,-323,401,-288,-678,143,100,-754,-196,742,-490,-816,679,152,-488,621,928,-340,-341,590,271,-103,391,673,428,-401,145,592,-559,410,960,-491,-543,299,240,-550,-345,565,138,-644,-144,359,-493,-78,-738,-70,835,-961,-807,16,1035,162,-2419,-292,3129,-79,-2504,1369,763,-1733,252,1238,720,59,-220,25,-34,-15,35,-490,145,1213,29,-1074,-436,-50,-624,40,1516,418,-916,-64,557,-164,182,679,-133,-348,609,1166,62,-455,530,105,-423,822,1161,182,106,156,469,416,-99,-299,190,347,570,332,-155,-68,-312,-135,513,700,298,134,318,-38,73,656,727,101,254,898,483,-127,-78,-7,411,644,290,-77,-198,52,255,271,90,-175,-222,425,766,80,-604,-339,-340,-562,360,945,-18,-974,-627,-387,-558,-66,79,-663,-1055,-591,-956,-1396,-904,-760,-1033,-1109,-896,-1276,-1883,-2047,-1719,-1735,-2053,-1805,-2166,-2915,-3255,-3341,-3304,-3022,-2917,-3551,-4193,-4724,-5179,-5371,-4805,-4020,-4557,-5431,-5612,-5736,-5620,-4103,-1935,-604,369,2669,4605,3121,554,1184,4184,6392,7648,8809,9002,6822,4719,5450,8027,9422],dtype=np.int16)
# fbank = compute_log_mel_fbank_fromsig(signal,16000,80)

# import random
# random.seed(100)
# test1 = np.ones(512,np.int16)
# for i in range(512):
#     test1[i] = random.randint(-300,300)
# compute_log_mel_fbank_fromsig(test1,16000,80)

