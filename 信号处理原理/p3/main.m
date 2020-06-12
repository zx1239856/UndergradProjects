[audio1, fs1] = audioread("data/en.mp3");
[audio2, fs2] = audioread("data/fr.mp3");
[audio3, fs3] = audioread("data/it.mp3");

if(fs1 ~= fs2 || fs1 ~= fs3)
    fprintf("Error! Audio files come with different sampling rates!\n");
    exit();
end

fs = fs1;

audio_len = max([length(audio1), length(audio2), length(audio3)]);

% zero-pad all audios to the same length
audio1 = [audio1', zeros(1,audio_len-length(audio1))];
audio2 = [audio2', zeros(1,audio_len-length(audio2))];
audio3 = [audio3', zeros(1,audio_len-length(audio3))];
% plot original audios wrt to t and f
plot_audio_t(audio1, audio2, audio3, 1);
plot_audio_f(audio1, audio2, audio3, fs, 2);

% modulate
N = audio_len;

fa1 = fft(audio1, N);
fa2 = fft(audio2, N);
fa3 = fft(audio3, N);

% mux
fres = zeros(1, 3 * N);

fres(N + 1 : 2 * N) = fa1;
fres(N / 2 + 1 : N) = fa2(1 : N / 2);
fres(2 * N + 1 : 5 * N / 2) = fa2(N / 2 + 1 : N);
fres(1 : N / 2) = fa3(1 : N / 2);
fres(5 * N / 2 + 1 : 3 * N) = fa3(N / 2 + 1 : N);
res = real(ifft(fres, 3 * N));

figure(3);
stem(0:3 * fs-1, abs(fft(res, 3 * fs)), '.');
figure(4);
stem(0:3 * N-1, res, '.');

% write to file
audiowrite('output.wav', res, fs * 3);


% load from saved file 
[res_in, fss] = audioread('output.wav');
demod = fft(res_in', 3 * N);
% demodulate
ra1 = demod(N + 1 : 2 * N);
ra2 = zeros(1, N);
ra3 = zeros(1, N);
ra2(1 : N / 2) = demod(N / 2 + 1 : N);
ra2(N / 2 + 1 : N) = demod(2 * N + 1 : 5 * N / 2);
ra3(1 : N / 2) = demod(1 : N / 2);
ra3(N / 2 + 1 : N) = demod(5 * N / 2 + 1 : 3 * N);
% restore
ra1 = real(ifft(ra1, N));
ra2 = real(ifft(ra2, N));
ra3 = real(ifft(ra3, N));
plot_audio_t(ra1, ra2, ra3, 5);
plot_audio_f(audio1, audio2, audio3, fs, 6);
% 
fprintf("Playing first audio ...\n");
soundsc(ra1, fs);
pause(length(ra1) / fs);
fprintf("Playing second audio ...\n");
soundsc(ra2, fs);
pause(length(ra2) / fs);
fprintf("Playing third audio ...\n");
soundsc(ra3, fs);
pause(length(ra3) / fs);


function plot_audio_t(a1, a2, a3, plot_num)
    figure(plot_num);
    t = 0 : max([length(a1), length(a2), length(a3)]) - 1;
    subplot(311);
    plot(t, a1, '-');
    subplot(312);
    plot(t, a2, '-');
    subplot(313);
    plot(t, a3, '-');
end

function plot_audio_f(a1, a2, a3, fs, plot_num)
    figure(plot_num);
    t = 0 : fs - 1;
    subplot(311);
    stem(t, abs(fft(a1, fs)), '.');
    subplot(312);
    stem(t, abs(fft(a2, fs)), '.');
    subplot(313);
    stem(t, abs(fft(a3, fs)), '.');
end