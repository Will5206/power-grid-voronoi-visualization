import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from scipy.spatial import Voronoi, Delaunay, ConvexHull
import math






def generate_power_grid_data(num_buses=30, grid_size=100, 
                            nominal_voltage=1.0, voltage_variation=0.05,
                            avg_connections=3, save_to_csv=True):
    """
    generate realistic power system bus voltage data for vis.
    
    params:
    num_buses : int
        num buses in the power grid
    grid_size : float
        size of the grid (coordinates will be from 0 to grid_size)
    nominal_voltage : float
        nominal voltage in per unit (p.u.)
    voltage_variation : float
        max variation from nominal voltage (+ - this value)
    avg_connections : int
        avg num of connections per bus
    save_to_csv : bool
        whether to save generated data to CSV files
    output:
    tuple
        (bus_data, connection_data) dfs containing the generated data
    """
    # structured power grid with more realistic geographic layout
     #divide the area into a grid-like structure
    rows = max(3, int(np.sqrt(num_buses / 2)))
    cols = max(3, int(np.sqrt(num_buses / 2)))
    
    # base grid of positions
    x_positions = np.linspace(0.1 * grid_size, 0.9 * grid_size, cols)
    y_positions = np.linspace(0.1 * grid_size, 0.9 * grid_size, rows)
    
    # random offset to each position to avoid perfect grid
    x_offsets = np.random.uniform(-0.3 * grid_size / cols, 0.3 * grid_size / cols, size=(rows, cols))
    y_offsets = np.random.uniform(-0.3 * grid_size / rows, 0.3 * grid_size / rows, size=(rows, cols))
    
    #positions - they'll follow streets, but with some variance
    positions = []
    for i in range(rows):
        for j in range(cols):
            # slight perturbation to grid positions
            x = x_positions[j] + x_offsets[i, j]
            y = y_positions[i] + y_offsets[i, j]
            



            if len(positions) < num_buses or random.random() < 0.8:
                positions.append((x, y))
                
                # sometimes add another position nearby (simulating substation clusters)
                if len(positions) < num_buses and random.random() < 0.2:
                    cluster_x = x + random.uniform(-0.1 * grid_size / cols, 0.1 * grid_size / cols)
                    cluster_y = y + random.uniform(-0.1 * grid_size / rows, 0.1 * grid_size / rows)
                    positions.append((cluster_x, cluster_y))


    while len(positions) < num_buses:
        idx = random.randint(0, len(positions) - 1)
        base_x, base_y = positions[idx]
        
        #  new position along a "street" (horizontal or vert line)
        if random.random() < 0.5:  #horizontal
            new_x = base_x + random.choice([-1, 1]) * random.uniform(0.05, 0.15) * grid_size
            new_y = base_y + random.uniform(-0.03, 0.03) * grid_size
        else:  # vertical
            new_x = base_x + random.uniform(-0.03, 0.03) * grid_size
            new_y = base_y + random.choice([-1, 1]) * random.uniform(0.05, 0.15) * grid_size
        
        # position w/in grid bounds
        new_x = max(0, min(grid_size, new_x))
        new_y = max(0, min(grid_size, new_y))
        
        positions.append((new_x, new_y))
    
    #num_buses positions
    positions = positions[:num_buses]
    
    # graph with buses at these positions
    G = nx.Graph()
    


    for i in range(num_buses):
        G.add_node(i)
    
    #  edges (connections) that prioritize Manhattan-distance connections ------------ simulates how power lines often follow streets
    for i in range(num_buses):
        # manhattan dist to all other buses
        distances = []
        for j in range(num_buses):
            if i != j:
                # manhat distance: |x1-x2| + |y1-y2|
                manhattan_dist = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                #also consider Euclidean distance to avoid unrealistic long connections
                euclidean_dist = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                

                combined_dist = 0.7 * manhattan_dist + 0.3 * euclidean_dist
                distances.append((j, combined_dist))
        
        distances.sort(key=lambda x: x[1])




        # connect to closest buses
        num_connections = min(random.randint(2, avg_connections + 1), len(distances))
        for k in range(num_connections):
            j, _ = distances[k]
            G.add_edge(i, j)
    
    # making sure the the graph is connected
    if not nx.is_connected(G):

        components = list(nx.connected_components(G))
        

        for i in range(len(components) - 1):
            component1 = list(components[i])
            component2 = list(components[i + 1])
            
            min_dist = float('inf')
            closest_pair = None

            
            for node1 in component1:
                for node2 in component2:
                    dist = abs(positions[node1][0] - positions[node2][0]) + abs(positions[node1][1] - positions[node2][1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)
            
            if closest_pair:
                G.add_edge(closest_pair[0], closest_pair[1])
    
    # voltages w spatial correlation
    voltages = {}
    
    # a few "source" buses with higher voltagess
    source_buses = random.sample(list(G.nodes()), min(5, num_buses))
    for node in source_buses:
        voltages[node] = nominal_voltage + random.uniform(0.5*voltage_variation, voltage_variation)
    
    # calc voltages for remaining buses based on distance from sources
    for node in G.nodes():
        if node not in voltages:
            source_distances = []
            for source in source_buses:
                manhattan_dist = abs(positions[node][0] - positions[source][0]) + abs(positions[node][1] - positions[source][1])
                source_distances.append((source, manhattan_dist))
            source_distances.sort(key=lambda x: x[1])
            closest_source, distance = source_distances[0]
            
            # V decays with distance from source
            normalized_distance = distance / (grid_size * 2)  # normalize by grid size
            voltage_drop = normalized_distance * 0.2  # up to 20% voltage drop at max dist
            
            #variation
            random_component = random.uniform(-voltage_variation/2, voltage_variation/2)
            voltages[node] = voltages[closest_source] * (1 - voltage_drop) + random_component
    


    # bus data df
    bus_data = []
    for node in G.nodes():
        x, y = positions[node]
        voltage = voltages[node]
            
        bus_data.append({
            'bus_id': node + 1,  # 1-indexed bus IDs
            'voltage': voltage,
            'x_coord': x,
            'y_coord': y
        })
    
    bus_df = pd.DataFrame(bus_data)
    
    # create connection data df
    connection_data = []
    for edge in G.edges():

        connection_data.append({ # both dirs for quicker lookup
            'from_bus': edge[0] + 1,  #  1 indexed bus IDs
            'to_bus': edge[1] + 1
        })
        connection_data.append({
            'from_bus': edge[1] + 1,
            'to_bus': edge[0] + 1
        })
    
    connection_df = pd.DataFrame(connection_data)
    
    # data to csv
    if save_to_csv:
        bus_df.to_csv('bus_voltage_data.csv', index=False)
        connection_df.to_csv('bus_connections.csv', index=False)
    
    return bus_df, connection_df

def visualize_grid(bus_df, connection_df, show_voltages=True):
    """
    to make a simple vis of the power grid with voltage levels.
    
    params:
    bus_df : df
        df containing bus data
    connection_df : df
        df containing connection data
    show_voltages : bool
         color nodes by voltage?
    """
    plt.figure(figsize=(12, 10))
    

    G = nx.Graph()
    

    for _, row in bus_df.iterrows():
        G.add_node(row['bus_id'], 
                  pos=(row['x_coord'], row['y_coord']),
                  voltage=row['voltage'])
    

    #  edges
    unique_connections = connection_df.drop_duplicates(subset=['from_bus', 'to_bus'])
    for _, row in unique_connections.iterrows():
        G.add_edge(row['from_bus'], row['to_bus'])
    
    #node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    #  voltage vals for coloring
    if show_voltages:
        voltages = list(nx.get_node_attributes(G, 'voltage').values())
        vmin = min(voltages)
        vmax = max(voltages)
    else:
        voltages = None




    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=200, 
                                 node_color=voltages if show_voltages else 'skyblue',
                                 cmap=plt.cm.viridis if show_voltages else None,
                                 vmin=vmin if show_voltages else None, 
                                 vmax=vmax if show_voltages else None)
    
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # colorbar
    if show_voltages:
        plt.colorbar(nodes, label='Voltage (p.u.)')
    
    plt.title('Power Grid Topology with Bus Voltages')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('power_grid_visualization.png')
    plt.show()

# orig voronoi helper funcs -------- keeping for reference
def compute_perpendicular_bisector(p1, p2):
    """
    calc the perpendicular bisector between two points.
    
    params:
    p1 : tuple
        Coordinates of the first point (x1, y1)
    p2 : tuple
        Coordinates of the second point (x2, y2)
output:
    tuple
        (slope, intercept) of the perpendicular bisector line
        If the line is vertical, returns (float('inf'), x-intercept)
    """
    # midpt of the two points
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    
    # check if the line connecting p1 and p2 is vert
    if p2[0] - p1[0] == 0:
        # ppb is horizontal
        return 0, mid_y
    
    # slope of line
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    
    # slope of the ppb
    if slope == 0:
        return float('inf'), mid_x
    else:
        perp_slope = -1 / slope
    
    # y-intercept using point-slope form:y - y1 = m(x - x1)
    # also y = mx - mx1 + y1 or y = mx + b where b = -mx1 + y1
    intercept = mid_y - perp_slope * mid_x
    return perp_slope, intercept

def line_intersection(line1, line2):
    """
    to find the intersection point of two lines.
    """
    m1, b1 = line1
    m2, b2 = line2
    
    #special cases for vertical lines (infinite slope)
    if m1 == float('inf') and m2 != float('inf'):
        x = b1 
        y = m2 * x + b2
        return (x, y)
    elif m2 == float('inf') and m1 != float('inf'):
        x = b2 
        y = m1 * x + b1
        return (x, y)
    elif m1 == float('inf') and m2 == float('inf'):
        # 2 vert lines ---- - either the same line or //
        return None
    
    # normal case -check if lines are //
    if m1 == m2:
        return None
    
    # solve syst of equatoins
    # y = m1*x + b1
    # y = m2*x + b2
    # ---> m1*x + b1 = m2*x + b2
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    
    return (x, y)

def calculate_circle_center(p1, p2, p3):
    """
    calculate the center of a circle passing through three points.
    this is equivalent to finding the circumcenter of a triangle.
    """
    # calc ppbs of two sides of the triangle
    bisector1 = compute_perpendicular_bisector(p1, p2)
    bisector2 = compute_perpendicular_bisector(p2, p3)
    #intersection of the ppbs
    return line_intersection(bisector1, bisector2)

def is_point_in_triangle(p, triangle):
    """
    check if a point is inside a triangle using barycentric coordinates.
    

    """
    p1, p2, p3 = triangle
    
    def area(x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    
    #  area of the original triangle
    A = area(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    
    #  areas of three triangles formed by the point and two vertices
    A1 = area(p[0], p[1], p2[0], p2[1], p3[0], p3[1])
    A2 = area(p1[0], p1[1], p[0], p[1], p3[0], p3[1])
    A3 = area(p1[0], p1[1], p2[0], p2[1], p[0], p[1])
    
    #if the sum of the three areas equals the original area then the point is inside
    return abs(A - (A1 + A2 + A3)) < 1e-9






# improved Voronoi algorithm implementation
def compute_voronoi_regions(bus_df, bounds=None):
    """
    compute Voronoi tessellation for power grid bus data using scipy.
    """
    # get points from df
    points = np.array([(row['x_coord'], row['y_coord']) for _, row in bus_df.iterrows()])
    
    # now to set default bounds if not provided
    if bounds is None:
        max_x = points[:, 0].max()
        max_y = points[:, 1].max()
        min_x = points[:, 0].min()
        min_y = points[:, 1].min()
        
        width = max_x - min_x
        height = max_y - min_y
        
        # margin
        margin = 0.1 * max(width, height)
        bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)



    min_x, min_y, max_x, max_y = bounds
    boundary_distance = 10 * max(max_x - min_x, max_y - min_y)  # large enough to contain all finite cells
    
    # four distant corner points
    boundary_points = np.array([
        [min_x - boundary_distance, min_y - boundary_distance],  #Bottom-left
        [max_x + boundary_distance, min_y - boundary_distance],  #Bottom-right
        [max_x + boundary_distance, max_y + boundary_distance],  #Top-right
        [min_x - boundary_distance, max_y + boundary_distance]   #Top-left
    ])
    
    # combine actual points with boundary points
    extended_points = np.vstack([points, boundary_points])
    
    #  calc v diagram for the extended point set
    vor = Voronoi(extended_points)
    
    #mapping from each bus_id to its Voronoi cell vertices
    voronoi_cells = {}
    
    # process each original point (excluding boundary points)
    for i in range(len(points)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        
        # skipingp empty regions or regions containing index -1
        if -1 in region or not region:
            continue


        vertices = [vor.vertices[j] for j in region]
        
        # clip the Voronoi cell to the specified bounds
        min_x, min_y, max_x, max_y = bounds
        
        # if any vertex is outside bounds
        needs_clipping = any(v[0] < min_x or v[0] > max_x or v[1] < min_y or v[1] > max_y for v in vertices)
        
        if needs_clipping:
            # clip vertices to bounds
            clipped_vertices = []
            for v in vertices:
                x, y = v
                x = max(min_x, min(max_x, x))
                y = max(min_y, min(max_y, y))
                clipped_vertices.append((x, y))
            
            vertices = clipped_vertices

        bus_id = bus_df.iloc[i]['bus_id']
        

        voronoi_cells[bus_id] = order_vertices_clockwise(vertices, points[i])
    


    for i in range(len(points)):
        bus_id = bus_df.iloc[i]['bus_id']
        


        if bus_id in voronoi_cells and voronoi_cells[bus_id]:
            continue
        
        # make a cell by hand for this point
        p = points[i]
        
        # note to self: this is a simplified approach --- for a better solution, compute the Delaunay triangulation and find all adjacent points
        tri = Delaunay(points)
        adjacent_points = set()
        
        # all triangles containing this point
        point_indices = np.where((tri.simplices == i).any(axis=1))[0]
        
        # now grab all points adjacent to this point in those triangles
        for idx in point_indices:
            for j in tri.simplices[idx]:
                if j != i:
                    adjacent_points.add(j)
        


        if not adjacent_points:
            distances = np.sqrt(np.sum((points - p)**2, axis=1))
            nearest_indices = np.argsort(distances)[1:4]  # Skip this point itself
            adjacent_points = set(nearest_indices)
        
        # v vertices from midpoints to adjacent points plus corners
        vertices = []
        for j in adjacent_points:
            midpoint = (p + points[j]) / 2
            vertices.append(midpoint)
        
        # corner points to ensure the cell is closed
        min_x, min_y, max_x, max_y = bounds
        corners = np.array([
            [min_x, min_y],  # Bottom-left
            [max_x, min_y],  # Bottom-right
            [max_x, max_y],  # Top-right
            [min_x, max_y]   # Top-left
        ])
        
        for corner in corners:
            # corners that are closer to this point than any adjacent point
            is_closest = True
            for j in adjacent_points:
                if np.linalg.norm(corner - points[j]) < np.linalg.norm(corner - p):
                    is_closest = False
                    break
            
            if is_closest:
                vertices.append(corner)
        


        # order vertices clockwise around point
        if vertices:
            voronoi_cells[bus_id] = order_vertices_clockwise(vertices, p)
    
    return voronoi_cells

def order_vertices_clockwise(vertices, center):
    """
    Order vertices clockwise around a center point.

    """
    if not vertices:
        return []
    

    vertices = np.array(vertices)
    center = np.array(center)
    
    # angles from center to each vertex
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    
    # vertices by angle
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]
    


    return [(x, y) for x, y in sorted_vertices]

def visualize_voronoi(bus_df, connection_df, voronoi_cells):
    """
    vis v tessellation with power grid bus data.
    """
    plt.figure(figsize=(14, 12))
    
   #map from voltage to color
    voltages = bus_df['voltage'].values
    vmin = min(voltages)
    vmax = max(voltages)
    normalize = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis
    
    # draw v cells
    for bus_id, vertices in voronoi_cells.items():
        if not vertices:
            continue
        
        # voltage for this bus
        voltage = bus_df.loc[bus_df['bus_id'] == bus_id, 'voltage'].values[0]
        color = cmap(normalize(voltage))
        
        # bus type for edge style
        if 'bus_type' in bus_df.columns:
            bus_type = bus_df.loc[bus_df['bus_id'] == bus_id, 'bus_type'].values[0]
            


            if bus_type == 'Generation':
                # geb buses w thicker borders
                edgecolor = 'red'
                linewidth = 2.0
                alpha = 0.7
            elif bus_type == 'Transmission':
                # trans buses w medium borders
                edgecolor = 'black'
                linewidth = 1.5
                alpha = 0.6
            else:  # Load

                edgecolor = 'gray'
                linewidth = 0.5
                alpha = 0.5
        else:
            edgecolor = 'black'
            linewidth = 1.0
            alpha = 0.6
        
        # polygon for the v polygon
        polygon = patches.Polygon(vertices, closed=True, fill=True, 
                                 alpha=alpha, edgecolor=edgecolor, 
                                 linewidth=linewidth, facecolor=color)
        plt.gca().add_patch(polygon)
    



    for _, row in connection_df.drop_duplicates(subset=['from_bus', 'to_bus']).iterrows():
        from_bus = bus_df[bus_df['bus_id'] == row['from_bus']].iloc[0]
        to_bus = bus_df[bus_df['bus_id'] == row['to_bus']].iloc[0]
        
        if 'bus_type' in bus_df.columns:
            # Use different line styles based on the buses being connected
            from_type = from_bus['bus_type']
            to_type = to_bus['bus_type']
            
            # Transmission lines between different types of buses
            if (from_type == 'Generation' and to_type == 'Transmission') or \
               (from_type == 'Transmission' and to_type == 'Generation'):
                # Generation to transmission: thicker, more alpha
                line_style = 'k-'
                linewidth = 2.0
                alpha = 0.8
            elif (from_type == 'Transmission' and to_type == 'Transmission'):
                # Transmission backbone: medium thickness
                line_style = 'k-'
                linewidth = 1.5
                alpha = 0.7
            else:
                # Distribution lines: thinner
                line_style = 'k-'
                linewidth = 0.8
                alpha = 0.5
        else:
            line_style = 'k-'
            linewidth = 1.0
            alpha = 0.5
        
        plt.plot([from_bus['x_coord'], to_bus['x_coord']], 
                [from_bus['y_coord'], to_bus['y_coord']], 
                line_style, alpha=alpha, linewidth=linewidth)
    


    # bus locs w diff styles for diff bus types
    if 'bus_type' in bus_df.columns:
        # gen buses -larger squares
        gen_buses = bus_df[bus_df['bus_type'] == 'Generation']
        plt.scatter(gen_buses['x_coord'], gen_buses['y_coord'], 
                   c=gen_buses['voltage'], cmap=cmap, vmin=vmin, vmax=vmax,
                   s=150, marker='s', edgecolor='red', linewidth=2, zorder=12)
        

        # transmission buses -medium circles
        trans_buses = bus_df[bus_df['bus_type'] == 'Transmission']
        plt.scatter(trans_buses['x_coord'], trans_buses['y_coord'], 
                   c=trans_buses['voltage'], cmap=cmap, vmin=vmin, vmax=vmax,
                   s=100, edgecolor='black', linewidth=1.5, zorder=11)
        
        # load buses- smaller circles
        load_buses = bus_df[bus_df['bus_type'] == 'Load']
        sc = plt.scatter(load_buses['x_coord'], load_buses['y_coord'], 
                        c=load_buses['voltage'], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=50, edgecolor='gray', linewidth=1, zorder=10)
    else:
        sc = plt.scatter(bus_df['x_coord'], bus_df['y_coord'], 
                        c=bus_df['voltage'], cmap=cmap, 
                        s=100, edgecolor='black', zorder=10)
    


    # bus IDs as labels ---------  only for gen and trans to reduce clutter
    if 'bus_type' in bus_df.columns:
        for _, row in bus_df[bus_df['bus_type'].isin(['Generation', 'Transmission'])].iterrows():
            plt.text(row['x_coord'], row['y_coord'], str(int(row['bus_id'])), 
                    ha='center', va='center', fontsize=8, fontweight='bold', 
                    color='white', zorder=15)
    else:
        for _, row in bus_df.iterrows():
            plt.text(row['x_coord'], row['y_coord'], str(int(row['bus_id'])), 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
    
    # Add a colorbar
    cbar = plt.colorbar(sc, label='Voltage (p.u.)')
    
    # Add a legend for bus types
    if 'bus_type' in bus_df.columns:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(normalize(gen_buses['voltage'].mean())),
                  markersize=10, markeredgecolor='red', markeredgewidth=2, label='Generation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(normalize(trans_buses['voltage'].mean())),
                  markersize=8, markeredgecolor='black', markeredgewidth=1.5, label='Transmission'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(normalize(load_buses['voltage'].mean())),
                  markersize=6, markeredgecolor='gray', markeredgewidth=1, label='Load')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title('Power Grid Voronoi Tessellation with Bus Voltages')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('power_grid_voronoi.png')
    plt.show()

if __name__ == "__main__":
    # gen power grid data
    bus_df, connection_df = generate_power_grid_data(
        num_buses=30,          # num of buses
        grid_size=100,         #grid size for coordinates
        nominal_voltage=1.0,   # nominal voltage in per unit
        voltage_variation=0.1, # variation from nominal voltage
        avg_connections=3,     # avg connections per bus
        save_to_csv=True       # save to CSV files
    )
    




    print(f"generated data for {len(bus_df)} buses with {len(connection_df)//2} connections")
    print("\nbus data sample:")
    print(bus_df.head())
    print("\nconnection data sample:")
    print(connection_df.head())
    
    

    # V stats
    print("\nvoltage statistics:")
    print(bus_df['voltage'].describe())
    
    # vis the grid
    visualize_grid(bus_df, connection_df, show_voltages=True)
    
    # v tessellation using the improved algorithm
    print("\ncomputing v tessellation...")
    voronoi_cells = compute_voronoi_regions(bus_df)
    

    print("visualizing v tessellation...")
    visualize_voronoi(bus_df, connection_df, voronoi_cells)