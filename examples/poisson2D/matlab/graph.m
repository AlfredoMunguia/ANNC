range = linspace (0, 1, 50);
[X, Y] = meshgrid (range, range);
Z = laplaciano (X, Y);
surf (X, Y, Z);
