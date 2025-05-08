# power-grid-voronoi-visualization
Voronoi tessellation visualization for power grid voltage distributions

Features
Synthetic power grid generation with realistic topology
Voronoi tessellation visualization of bus voltages
Spatially correlated voltage calculations

Libraries
pip install numpy pandas networkx matplotlib scipy

Usage
To run the complete implementation: python main-with-voronoi.py

This generates:
- Synthetic power grid with 30 buses
- Basic grid visualization (power_grid_visualization.png)
- Voronoi tessellation (power_grid_voronoi.png)
- csv files with bus data and connections

Files
main-with-voronoi.py - complete implementation with Voronoi tessellation
*.csv - Generated bus voltage and connection data
*.png - Output visualizations

References
Based on: Lyons-Galante, I., et al. (2023). Alternatives to Contour Visualizations for Power Systems Data. arXiv:2308.09153.
