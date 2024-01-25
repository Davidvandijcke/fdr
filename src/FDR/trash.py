# use further removed points as jumpfrom and jumpto points

def explore(self, point, J_grid, visited_points=None, shift = 1):
    
    if visited_points is None:
        visited_points = set()
    
    neighbors = []

    for d in range(J_grid.ndim):
        neighbor = point.copy()
        if (shift == 1 and neighbor[d] < J_grid.shape[d] - 1) or \
            (shift == -1 and neighbor[d] > 0):
            neighbor[d] += shift
            if J_grid[tuple(neighbor)] == 0:
                visited_points.add(tuple(neighbor))
            neighbors.append(neighbor)
    
    # Check if all neighbors are jump points, if so, continue exploring
    if all(J_grid[tuple(neighbor)] == 1 for neighbor in neighbors):
        for neighbor in neighbors:
            if tuple(neighbor) not in visited_points:
                visited_points.update(self.explore(neighbor, J_grid, visited_points, shift = shift))
    
    return visited_points


def getNearestPoint(self, neighbors, jumpfrom, u):
    # jumpto point
    pointslist = [self.grid_x_og[tuple(neighbors[j])] if self.grid_x_og[tuple(neighbors[j])] != []  # if grid cell is empty, assign centerpoint
                    else [self.grid_x[tuple(neighbors[j])] + self.resolution / 2] for j in range(len(neighbors))]

    # get closest point
    dists = [[np.linalg.norm(jumpfrom - point) for point in pointslist[j]] for j in range(len(neighbors))]
    idx = np.argmin([np.argmin(sublist) for sublist in dists])
    closest = tuple(neighbors[idx])
    Yjumpto = u[closest]

    dest_points = self.grid_x_og[closest]
    if len(dest_points) == 0:
        dest_points = self.grid_x[closest] + self.resolution / 2
    dest_points = np.stack(dest_points).squeeze()

    if dest_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
        jumpto = np.mean(dest_points, axis = 0)
    else:
        jumpto = dest_points
        
    return jumpto, Yjumpto
                
def boundaryGridToData(self, J_grid, u, average = False):
    # get the indices of the J_grid where J_grid is 1


    k = np.array(np.where(J_grid == 1))
    

    # Store the average points
    Y_jumpfrom = []
    Y_jumpto = []
    Y_jumpsize = []
    Y_boundary = []
    X_jumpfrom = []
    X_jumpto = []


    # Iterate over the boundary points
    for i in range(k.shape[1]):

        origin_points = []
        dest_points = []

        # Get the coordinates of the current boundary point
        point = k[:, i]
        
        neighbors = []
        # Initialize a list to store the neighboring hypervoxels
        neighbors_right = list(self.explore(point, J_grid, shift = 1)) 
        neighbors_left = list(self.explore(point, J_grid, shift = -1))
        # for d in range(J_grid.ndim):
        #     neighbor = point.copy()
        #     if neighbor[d] < J_grid.shape[d] - 1:
        #         neighbor[d] += 1
        #         neighbors.append(neighbor)           

       
        # origin_points
        point_pixel = self.grid_x[tuple(point)] + self.resolution / 2
        jumpfrom, Yjumpfrom = self.getNearestPoint(neighbors_left, point, u)
        jumpto, Yjumpto = self.getNearestPoint(neighbors_right, point_pixel, u)


        # append to lists
        Y_boundary.append((jumpfrom + jumpto) / 2)
        Y_jumpfrom.append(Yjumpfrom)
        Y_jumpto.append(Yjumpto)
        Y_jumpsize.append(Yjumpto - Yjumpfrom)
        X_jumpfrom.append(jumpfrom)
        X_jumpto.append(jumpto)

    if Y_boundary:
        Y_boundary = np.stack(Y_boundary)
        Y_jumpfrom = np.stack(Y_jumpfrom)
        Y_jumpto = np.stack(Y_jumpto)
        Y_jumpsize = np.stack(Y_jumpsize)

        # create named array to return
        rays = [Y_boundary[:,d] for d in range(Y_boundary.shape[1])] + [Y_jumpfrom, Y_jumpto, Y_jumpsize]
        names = ["X_" + str(d) for d in range(Y_boundary.shape[1])] + ["Y_jumpfrom", "Y_jumpto", "Y_jumpsize"]
        jumps = np.core.records.fromarrays(rays, names=names)
    else:
        jumps = None

    return jumps