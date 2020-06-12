fprintf("Single tone sample:\n");
for i = 0 : 9
    [audio, fs] = audioread(sprintf('data/dtmf-%d.wav', i));
    cmp(audio, fs, true);
end

fprintf("Another synthesized sample:\n");
[audio, fs] = audioread('data/sample.mp3');
result = '??????????';
k = 1;
for i = 1:round(length(audio)/10):length(audio)
    tmp = audio(i : min(i + 8820 - 1, length(audio)));
    ret = cmp(tmp, fs, false);
    result(k) = ret;
    k = k + 1;
end
fprintf("Result: %s\n", result);

fprintf("Zhou Hongwei sample:\n");
[audio, fs] = audioread('data/zhw.wav');
audio = (audio(:, 1) + audio(:, 2)) ./ 2;
consensus = ''; fft_res = ''; goertzel_res = '';
interval = 1000;
for i = 0.2*fs:interval:length(audio) - interval
    ret_fft = fft_dtmf(audio(i:i+interval), fs);
    ret_goertzel = goertzel(audio(i:i+interval), fs);
    if(length(ret_fft) == 1 && ret_fft ~= '?' && ret_fft <= '9' && ret_fft >= '0')
        fft_res = append(fft_res, ret_fft);
    elseif(isempty(fft_res) || fft_res(length(fft_res)) ~= '?')
        fft_res = append(fft_res, '?');
    end
    if(length(ret_goertzel) == 1 && ret_goertzel ~= '?' && ret_goertzel <= '9' && ret_goertzel >= '0')
        goertzel_res = append(goertzel_res, ret_goertzel);
    elseif(isempty(goertzel_res) || goertzel_res(length(goertzel_res)) ~= '?')
        goertzel_res = append(goertzel_res, '?');
    end
    ret = '?';
    if(ret_goertzel == ret_fft)
        ret = ret_goertzel;
    end
    if(ret ~= '?' && ret <= '9' && ret >= '0')
        consensus = append(consensus, ret);
    elseif(isempty(consensus) || consensus(length(consensus)) ~= '?')
        consensus = append(consensus, '?');
    end
end
fprintf("Zhou Hongwei's phone number[consensus] is %s\nOriginal seq: %s\n", collator(consensus, 3), consensus);
fprintf("Zhou Hongwei's phone number[FFT] is %s\nOriginal seq: %s\n", collator(fft_res, 4), fft_res);
fprintf("Zhou Hongwei's phone number[Goertzel] is %s\nOriginal seq: %s\n", collator(goertzel_res, 4), goertzel_res);

function res = collator(temp_res, thrs)
    res = '';
    freq = 1;
    for i = 2 : length(temp_res)
        curr = temp_res(i);
        if(curr ~= temp_res(i-1))
            if(freq >= thrs)
                res = append(res, temp_res(i - 1));
            end
            freq = 0;
        end
        freq = freq + 1;
    end
end

function ret = cmp(audio, fs, output)
    t = clock;
    res1 = fft_dtmf(audio, fs);
    elp = etime(clock, t) * 1000;
    if(output)
        fprintf("[FFT]      Result keycode: %s, time: %f ms\n", res1, elp);
    end
    t = clock;
    res2 = goertzel(audio, fs);
    elp = etime(clock, t) * 1000;
    if(output)
        fprintf("[Goertzel] Result keycode: %s, time: %f ms\n", res2, elp);
    end
    if(res1 == res2)
        ret = res1;
    else
        ret = '?';
    end
end