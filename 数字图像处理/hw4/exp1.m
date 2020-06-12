img = rgb2gray(imread('lena.png'));

% ratios = [1; 2; 4; 8];  % 1/4, 1/16, 1/64
ratios = 8;
blk_sizes = 8;
% blk_sizes = [8,16,32,64,128,256,512];

img_size = size(img);
h = img_size(1);
w = img_size(2);

for m = 1:length(blk_sizes)
    blk_size = blk_sizes(m);
    h_blks = h / blk_size;
    w_blks = w / blk_size;

    for k=1:length(ratios)
        ratio = ratios(k);
        % 2D-DCT try
        out_dct_2d = blk_transform(img, blk_size, @(x)dct_2d(x, ratio));
        out_idct = uint8(blk_transform(out_dct_2d, blk_size, @idct_2d));
        % 1D-DCT try
        out_dct_1d = blk_transform(img, blk_size, @(x)dct_1d(x, ratio));
        out_idct_1d = uint8(blk_transform(out_dct_1d, blk_size, @idct_2d));
        
        name = ['Blk_size: ', num2str(blk_size),', Ratio: 1/',num2str(ratio * ratio)];
        figure('name', ['2D,', name]), title(name), imshow(out_idct);
        figure('name', ['1D,', name]), title(name), imshow(out_idct_1d);
        p = psnr(out_idct, img);
        p2 = psnr(out_idct_1d, img);
        fprintf("Blk_size: %d, Ratio: 1/%d, [2D]PSNR: %f, [1D]PSNR: %f\n", ...
            blk_size, ratio * ratio, p, p2);
    end
end

function out = dct_2d(blk, ratio)
    blk = double(blk) - 128;
    in_size = size(blk);
    out = zeros(in_size);
    h = in_size(1) / ratio;
    w = in_size(2) / ratio;
    res = dct2(blk);
    out(1:h, 1:w) = res(1:h, 1:w);
end

function out = dct_1d(blk, ratio)
    blk = double(blk) - 128;
    in_size = size(blk);
    h = in_size(1) / ratio;
    w = in_size(2) / ratio;
    rwise = dct(blk')';
    tmp = zeros(size(rwise));
    tmp(:, 1:w) = rwise(:, 1:w);
    cwise = dct(tmp);
    out = zeros(size(blk));
    out(1:h, :) = cwise(1:h, :);
end

function out = idct_2d(blk)
    out = round(idct2(blk) + 128);
end

function out = blk_transform(img, blk_size, trans_func)
    in_size = size(img);
    out = zeros(in_size);
    h_blks = in_size(1) / blk_size;
    w_blks = in_size(2) / blk_size;
    for i=1:h_blks
        for j=1:w_blks
            out((i-1)*blk_size+1:i*blk_size, (j-1)*blk_size+1:j*blk_size) ...
                = trans_func(img((i-1)*blk_size+1:i*blk_size, (j-1)*blk_size+1:j*blk_size));
        end
    end
end