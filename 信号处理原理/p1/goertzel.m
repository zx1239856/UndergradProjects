function ret = goertzel(audio, fs)
    keys = ['1', '2', '3', 'A'; '4', '5', '6', 'B'; '7', '8', '9', 'C';
        '*', '0', '#', 'D'];
    freq = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
    N = length(audio);
    freq = N * freq / fs;  % N * f / fs
    P = zeros(1, length(freq));
    for idx = 1 : length(freq) % iterate over all freqs
        C = 2 * cos(2 * pi * freq(idx) / N);
        Q1 = 0; Q2 = 0;
        for i = 3 : N  % DP
            Q0 = C * Q1 - Q2 + audio(i);
            Q2 = Q1;
            Q1 = Q0;
        end
        P(idx) = Q1^2 + Q2^2 - C * Q1 * Q2;
    end
    P = abs(P);
    row = P(1:4) == max(P(1:4));
    col = P(5:8) == max(P(5:8));
    ret = keys(row, col);
end
