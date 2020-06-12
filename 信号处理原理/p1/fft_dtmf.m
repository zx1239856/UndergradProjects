function ret = fft_dtmf(audio, fs)
    keys = ['1', '2', '3', 'A'; '4', '5', '6', 'B'; '7', '8', '9', 'C';
        '*', '0', '#', 'D'];
    row_freq = [697, 770, 852, 941];
    col_freq = [1209, 1336, 1477, 1633];
    res = abs(fft(audio, fs));
    row = find(res(650:1000) == max(res(650:1000))) + 650;
    col = find(res(1150:1700) == max(res(1150:1700))) + 1149;
    rerr = abs(row_freq - row);
    cerr = abs(col_freq - col);
    row = rerr == min(rerr);
    col = cerr == min(cerr);
    ret = keys(row, col);
end

