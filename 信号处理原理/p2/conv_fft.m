function res = conv_fft(x, y, optimize)
    M = length(x);
    N = length(y);
    L = M + N - 1;
    if(optimize)
        L = 2 ^ nextpow2(L);
    end
    x = [x, zeros(1, L - M + 1)];
    y = [y, zeros(1, L - N + 1)];
    res = ifft(fft(x) .* fft(y));
    res = res(1:M + N - 1);
end