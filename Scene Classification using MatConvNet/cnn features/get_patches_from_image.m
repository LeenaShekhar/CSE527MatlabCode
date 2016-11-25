function [patches] = get_patches_from_image(image, ph)

[imh, imw, k] = size(image);

patch_ul = image(1:ph, 1:ph, :);
patch_ur = image(1:ph, imw-ph+1:imw, :);
patch_ll = image(imh-ph+1:imh, 1:ph, :);
patch_lr = image(imh-ph+1:imh, imw-ph+1:imw, :);
patch_c = image(imh/2-ph/2+1:imh/2+ph/2, imw/2-ph/2+1:imw/2+ph/2, :);

image = flipdim(image ,2); 

patch_inv_ul = image(1:ph, 1:ph, :);
patch_inv_ur = image(1:ph, imw-ph+1:imw, :);
patch_inv_ll = image(imh-ph+1:imh, 1:ph, :);
patch_inv_lr = image(imh-ph+1:imh, imw-ph+1:imw, :);
patch_inv_c = image(imh/2-ph/2+1:imh/2+ph/2, imw/2-ph/2+1:imw/2+ph/2, :);

patches = single(cat(4, patch_ul, patch_ur, patch_ll, patch_lr, patch_c, patch_inv_ul, patch_inv_ur, patch_inv_ll, patch_inv_lr, patch_inv_c)) ;



     




