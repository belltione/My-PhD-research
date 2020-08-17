function [ToPoImage]=TopoImage_Sheng(ImageValue,Coordinate,GRID_SCALE,imageshow)

if nargin < 3
    GRID_SCALE = 32;        % plot map on a 32X32 grid
    imageshow=1;
end
if length(ImageValue)~=size(Coordinate,1)
    error('Please check the input value and the channel location information, the length of them is different.')
end

% rmax = 0.5;
rmax=max(sqrt(sum(Coordinate.^2,2)));

X_coor = Coordinate(:,1); 
Y_coor = Coordinate(:,2); 
xi = X_coor;   % x-axis description (row vector)
yi = Y_coor;   % y-axis description (row vector)
plotrad = rmax;
squeezefac = rmax/plotrad;
unsh = (GRID_SCALE+1)/GRID_SCALE;
intx=xi;
inty=yi;
intx = intx*squeezefac;   
inty = inty*squeezefac;  
xmin = min(-rmax,min(intx)); xmax = max(rmax,max(intx));
ymin = min(-rmax,min(inty)); ymax = max(rmax,max(inty));
xi = linspace(xmin,xmax,GRID_SCALE);   % x-axis description (row vector)
yi = linspace(ymin,ymax,GRID_SCALE);   % y-axis description (row vector)
[Xi,Yi,ZiC] = griddata(inty,intx,double(ImageValue),yi',xi,'v4'); 

mask = (sqrt(Xi.^2 + Yi.^2) <= rmax); % mask outside the plotting circle
ZiC(find(mask == 0)) = NaN;                         % mask non-plotting voxels with NaNs
ToPoImage=ZiC';
if imageshow==1
    hold on
    surface(Xi*unsh,Yi*unsh,zeros(size(ToPoImage)),ToPoImage,'EdgeColor','none')
    set(gca,'Xlim',[-rmax rmax]*1,'Ylim',[-rmax rmax]*1)
end
