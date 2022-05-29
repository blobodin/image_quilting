import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def minimal_cost_path(B1, B2, block_dim, overlap):
    # Assume looking at column overlap
    B1_ov = B1[:, (block_dim[1] - overlap):]
    B2_ov = B2[:, :overlap]

    # Dynamic Programming for cut
    error_surface = np.power(B1_ov - B2_ov, 2)
    dp = np.zeros(error_surface.shape)
    paths = []
    for i in range(dp.shape[0]):
        for j in range(dp.shape[1]):
            dp[i, j] = error_surface[i, j]

            if i == 0:
                paths.append([])
            else:
                min_list = [(dp[i - 1, j], j)]
                if j - 1 >= 0:
                    min_list.append((dp[i - 1, j - 1], j - 1))
                if j + 1 < dp.shape[1]:
                    min_list.append((dp[i - 1, j + 1], j + 1))

                prev_min = min(min_list)
                dp[i, j] += prev_min[0]

                paths[j].append((i - 1, prev_min[1]))

    # Find min cost path
    min_index = 0
    for i in len(paths):
        if dp[-1, i] < dp[-1, min_index]:
            min_index = i

    min_cost_path = paths[min_index]
    min_cost_path.append((dp.shape[0] - 1, min_index))

    return np.matrix(min_cost_path)

def calc_error(B1, B2):
    return np.linalg.norm(B1 - B2)

def get_row_overlap_regions(neighbor, blocks, block_dim, overlap)
    # Get row overlap region from already placed neighbor
    neighbor_row_region = neighbor[(block_dim[0] - overlap):, :]

    # Get row overlap regions of candidate texture blocks
    block_row_regions = blocks[:, :, :overlap, :]

    # Insert dim if needed!

    return neighbor_row_regions, block_row_regions

def get_col_overlap_regions(neighbor, blocks, block_dim, overlap):
    # Get col overlap region from already placed neighbor
    neighbor_col_region = neighbor[:, (block_dim[1] - overlap):]

    # Get col overlap regions of candidate texture blocks
    block_col_regions = blocks[:, :, :, :overlap]

    # Insert dim if needed!

    return neighbor_col_region, block_col_regions

def get_valid_blocks(neighbor_overlap, block_overlaps, error_tol):
    # Calculate L2 norm between overlaps
    error_matrix = np.linalg.norm(block_overlaps - neighbor_overlap, axis=(2, 3))

    # Get boolean matrix of blocks whose error overlap satifies the error tolerance
    return error_matrix < error_tol

def find_valid_block(row_neighbor, col_neighbor, texture, block_dim, overlap, error_tol):
    # Get all candidate inserts from base texture
    blocks = sliding_window_view(texture, block_dim)

    # Prepare boolean matrix that will designate blocks that satisfy error
    # constraints in row + col overlap regions
    valid_blocks = np.zeros((blocks.shape[0], blocks.shape[1]))

    # If testing error tolerance horizontally
    if row_neighbor != None:
        # Get row overlap regions of neighbor + blocks
        neighbor_row_region, block_row_regions = get_row_overlap_regions(row_neighbor, blocks, block_dim, overlap)

        # Get boolean matrix of blocks that satisfy error tolerance in row
        # overlap region
        row_valid_blocks = get_valid_blocks(neighbor_row_region, block_row_regions, error_tol)

        # Adjust overall boolean matrix with new info
        valid_blocks *= row_valid_blocks

    # If testing error tolerance vertically
    if col_neighbor != None:
        # Get col overlap regions of neighbor + blocks
        neighbor_col_region, block_col_regions = get_col_overlap_regions(col_neighbor, blocks, block_dim, overlap)

        # Get boolean matrix of blocks that satisfy error tolerance in col
        # overlap region
        col_valid_blocks = get_valid_blocks(neighbor_col_region, block_col_regions, error_tol)

        # Adjust overall boolean matrix with new info
        valid_blocks *= col_valid_blocks

    # Get all indexes of blocks whose overlap satify the error tolerance
    valid_block_indexes = np.argwhere(valid_blocks)

    # Randomly choose index of valid block
    rnd_choice = np.random.choice(range(len(valid_block_indexes)))
    chosen_block_index = valid_block_indexes[rnd_choice]

    # Grab randomly chosen block
    chosen_block = blocks[chosen_index[0], chosen_index[1], :, :]

    return chosen_block

def quilt(texture, output_dim, block_dim, overlap, error_tol):

    output = np.zeros(output_dim)

    row_step = block_dim[0] - row_overlap
    col_step = block_dim[1] - col_overlap
    for i in range(0, output_dim[0], row_step):
        for j in range(0, output_dim[1], col_step):
            row_neighbor = None
            col_neighbor = None

            if i != 0:
                row_neighbor = output[(i - row_step):(i - row_step + block_dim[0]), j:(j + block_dim[1])]
            if j != 0:
                col_neighbor = output[i:(i + block_dim[0]), (j - col_step):(j - col_step + block_dim[1])]

            # Get Valid Block to Insert
            new_block = find_valid_block(row_neighbor, col_neighbor, texture, block_dim, overlap, error_tol)

            # Compute Min Cost Path
            h_cut = np.flip(minimal_cost_path(row_neighbor.T, new_block.T, block_dim, overlap), axis=1)
            v_cut = minimal_cost_path(col_neighbor, new_block, block_dim, overlap)

            # Insert Block


    return
