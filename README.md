This repository contains MATLAB scripts for generating different geometric patterns, including a circle, a diamond shape, and a chessboard pattern. Each script uses simple logic to produce the desired shape on a 2D grid.

Code Descriptions

1. Circle

File: circle_pattern.m

Description:
This script creates a circle on a 100x100 grid using the Euclidean distance formula. Pixels within the given radius from the circle's center are marked, forming a circular shape.

Key Parameters:

Cx, Cy: Center coordinates of the circle.

Radius: Radius of the circle.

Logic:
For each pixel , the Euclidean distance from the center is calculated as:



If the distance is less than or equal to the radius, the pixel is marked as part of the circle.

Usage: Run the script to visualize a white circle on a black background.

2. Diamond (City Block Distance)

File: diamond_pattern.m

Description:
This script creates a diamond shape on a 100x100 grid using the City Block Distance (Manhattan Distance) metric. The diamond is centered at the specified coordinates and has a defined radius.

Key Parameters:

Cx, Cy: Center coordinates of the diamond.

Radius: Distance from the center to the farthest point in the diamond.

Logic:
For each pixel , the City Block Distance is calculated as:



If the distance is less than or equal to the radius, the pixel is marked as part of the diamond.

Usage: Run the script to visualize a white diamond shape on a black background.

3. Chessboard

File: chessboard_pattern.m

Description:
This script generates a chessboard-like pattern on a 100x100 grid. The pattern alternates between black and white squares.

Key Parameters:

Grid size: The resolution of the grid is fixed at 100x100.

Square size: Each square in the chessboard has a size of 10x10 pixels.

Logic:
For each pixel , the square color is determined using:



If the result is 0, the square is white; otherwise, it is black.

Usage: Run the script to visualize a chessboard pattern with alternating black and white squares.

Requirements

MATLAB (any recent version)
