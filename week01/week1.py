import torch
import math
import torch.nn.functional as F
import torchaudio
import numpy as np

def my_stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None):
    if not win_length:
        win_length = n_fft
    if not hop_length:
        hop_length = math.floor(win_length / 4)
    if not window:
        window = torch.ones(win_length)
        print(window.size)
    if window.size()[-1] < n_fft:
        pad_size_left = math.floor((n_fft - window.size[-1]) / 2)
        pad_size_right = math.ceil((n_fft - window.size[-1]) / 2)
        window = F.pad(window, (pad_size_left, pad_size_right), mode = "reflect")
    # target size m = кол-во sliding window, n_ftt, win_length
    #print(window.shape, win_length, input[: win_length].shape)
    stft_arr = []
    #print(input.shape[-1])
    num_rows = (1 + n_fft/2)
    k = 2 * np.pi * torch.outer(torch.arange(num_rows), torch.arange(win_length)) / win_length
    # print(k.shape, torch.arange(n_fft).shape)
    exp_m = torch.stack((torch.cos(k), -torch.sin(k)))
    for m in range(math.ceil((input.shape[-1]) / hop_length)):
        data = input[m*hop_length : m*hop_length + win_length]
        if data.shape[0] == win_length:
            # print(data.shape, win_length)
            # data =torch.nn.functional.pad(data, (0, -data.shape[0] + win_length))
            # print(data.shape)
        #print("first mul", (input[m*hop_length : m*hop_length + win_length] * exp_m).shape)
            stft = window * data * exp_m
            stft_arr.append(stft.sum(dim=2).T)
    stft = torch.stack(stft_arr, axis=1)
    #print(stft.shape)
    return stft
y = torchaudio.load("fff163132_3_7778.flac")
print(y[0][0].shape)
b = my_stft(y[0][0], 100)
a = torch.stft(y[0][0], n_fft = 100, return_complex = False, center = False)
print("target", a.shape, "result", b.shape)
print("target", a[1, :10], "result", b[1, :10])
c = (a-b).numpy()
print(np.where(abs(c)>1e-4, 1, 0).sum())
print(torch.allclose(a,b, atol = 1e-4,  equal_nan=True))

