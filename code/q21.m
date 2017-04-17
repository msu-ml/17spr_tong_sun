% draw scatter plot for (0, 0), (−1, 2), (−3, 6), (1, −2), (3, −6)
points = [0 0; -1 2; -3 6; 1 -2; 3 -6];
x = points(:,1);
y = points(:,2);

scatter(x, y, '*');
xlim([-4, +4]);
ylim([-7, +7]);
