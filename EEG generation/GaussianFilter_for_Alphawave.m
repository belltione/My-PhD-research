load('Electrode_coordinate.mat')

sigma = 80;
gaussian_weight = exp( -((Coor(:,1) - Coor(25,1)).^2+(Coor(:,2) - Coor(25,2)).^2)/(2*sigma^2) );
gaussian_weight = gaussian_weight';

save gaussian_weight gaussian_weight