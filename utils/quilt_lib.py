import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import copy

from pathlib import Path
from typing import Tuple, Union
from PIL import Image

#------------------------------------------------------------------------------
# GENERIC PATCH/BLOCK FUNCTIONS
#------------------------------------------------------------------------------

def get_block_set(texture, block_dim):
    # Get all candidate inserts from base texture
    blocks = sliding_window_view(texture, block_dim)

    # Reshape blocks matrix appropriately
    blocks = np.reshape(blocks, (blocks.shape[0], blocks.shape[1], blocks.shape[3], blocks.shape[4], blocks.shape[5]))

    return blocks

def insert_block(quilt, x, y, block, block_dim, h_cut, v_cut):
    h_wipe = np.zeros(block.shape)
    v_wipe = np.zeros(block.shape)

    # Copy for visualizing
    # new_block = np.copy(block)

    if h_cut is not None:
        for cut_loc in h_cut:
            (i, j) = cut_loc

            h_wipe[:i, j, :] = 1
            block[:i, j, :] = 0

            # new_block[i, j, 0] = 1
            # new_block[i, j, 1] = 0
            # new_block[i, j, 2] = 0

    if v_cut is not None:
        for cut_loc in v_cut:
            (i, j) = cut_loc

            v_wipe[i, :j, :] = 1
            block[i, :j, :] = 0

            # new_block[i, j, 0] = 1
            # new_block[i, j, 1] = 0
            # new_block[i, j, 2] = 0

    wipe = h_wipe + v_wipe > 0

    # Cut wipe and block if at right edge or bottom boundary before inserting
    quilt[x:(x + block_dim[0]), y:(y + block_dim[1]), :] *= wipe
    quilt[x:(x + block_dim[0]), y:(y + block_dim[1]), :] += block

    # write_image("cut.png", new_block)

    return quilt

def find_valid_quilt_block(row_neighbor, col_neighbor, blocks, block_dim, overlap):
    valid_block_indexes = []

    # Prepare error matrix to determine sampling
    error_matrix = np.zeros((blocks.shape[0], blocks.shape[1]))
    flag = False

    # If testing error tolerance horizontally
    if row_neighbor is not None:
        # Cut blocks to match neighbor dimensions
        row_blocks = blocks[:, :, :row_neighbor.shape[0], :row_neighbor.shape[1], :]

        # Get row overlap regions of neighbor + blocks
        neighbor_row_region, block_row_regions = get_row_overlap_regions(row_neighbor, row_blocks, block_dim, overlap)

        # Compute error matrix and add to total error matrix
        row_error_matrix = get_error_matrix(neighbor_row_region, block_row_regions)
        error_matrix += row_error_matrix
        flag = True

    # If testing error tolerance vertically
    if col_neighbor is not None:
        # Cut bocks to match neighbor dimensions
        col_blocks = blocks[:, :, :col_neighbor.shape[0], :col_neighbor.shape[1], :]

        # Get col overlap regions of neighbor + blocks
        neighbor_col_region, block_col_regions = get_col_overlap_regions(col_neighbor, col_blocks, block_dim, overlap)

        # Compute error matrix and add to total error matrix
        col_error_matrix = get_error_matrix(neighbor_col_region, block_col_regions)
        error_matrix += col_error_matrix
        flag = True

    if not flag:
        # If we didnt use row or col neighbor, all blocks have same error
        error_matrix += 1

    # Get error tolerance
    t = get_error_t(error_matrix)

    # Valid blocks meet error tolerance t
    valid_blocks = error_matrix <= t

    # Get all indexes of blocks whose overlap satify the error tolerance
    valid_block_indexes = np.argwhere(valid_blocks)

    # Randomly choose index of valid block
    rnd_choice = np.random.choice(range(len(valid_block_indexes)))
    chosen_block_index = valid_block_indexes[rnd_choice]

    # Grab randomly chosen block
    chosen_block = blocks[chosen_block_index[0], chosen_block_index[1], :, :, :]

    return np.copy(chosen_block)

def find_valid_transfer_block(row_neighbor, col_neighbor, blocks, block_dim, overlap, curr_region, c_blocks, curr_c_target, alpha):
    valid_block_indexes = []

    # APrepare error matrix to determine sampling
    error_matrix = np.zeros((blocks.shape[0], blocks.shape[1]))
    flag = False

    # If testing error tolerance horizontally
    if row_neighbor is not None:
        # Cut blocks to match neighbor dimensions
        row_blocks = blocks[:, :, :row_neighbor.shape[0], :row_neighbor.shape[1], :]

        # Get row overlap regions of neighbor + blocks
        neighbor_row_region, block_row_regions = get_row_overlap_regions(row_neighbor, row_blocks, block_dim, overlap)

        # Compute error matrix and add to total error matrix
        row_error_matrix = get_error_matrix(neighbor_row_region, block_row_regions)
        error_matrix += row_error_matrix
        flag = True

    # If testing error tolerance vertically
    if col_neighbor is not None:
        # Cut bocks to match neighbor dimensions
        col_blocks = blocks[:, :, :col_neighbor.shape[0], :col_neighbor.shape[1], :]

        # Get col overlap regions of neighbor + blocks
        neighbor_col_region, block_col_regions = get_col_overlap_regions(col_neighbor, col_blocks, block_dim, overlap)

        # Compute error matrix and add to total error matrix
        col_error_matrix = get_error_matrix(neighbor_col_region, block_col_regions)
        error_matrix += col_error_matrix
        flag = True

    if not flag:
        # If we didnt use row or col neighbor, all blocks have same error
        error_matrix += 1

    if curr_region is not None:
        # Check correspondence with previous iteration
        error_matrix += get_error_matrix(curr_region, blocks[:, :, :curr_region.shape[0], :curr_region.shape[1], :])

    # Get error between blocks and correspondence map
    c_error_matrix = get_error_matrix(curr_c_target, c_blocks[:, :, :curr_c_target.shape[0], :curr_c_target.shape[1]])
    error_matrix = alpha * error_matrix + (1 - alpha) * c_error_matrix

    # Get error tolerance
    t = get_error_t(error_matrix)

    # Valid blocks meet error tolerance t
    valid_blocks = error_matrix <= t

    # Get all indexes of blocks whose overlap satify the error tolerance
    valid_block_indexes = np.argwhere(valid_blocks)

    # Randomly choose index of valid block
    rnd_choice = np.random.choice(range(len(valid_block_indexes)))
    chosen_block_index = valid_block_indexes[rnd_choice]

    # Grab randomly chosen block
    chosen_block = blocks[chosen_block_index[0], chosen_block_index[1], :, :, :]

    return np.copy(chosen_block)

#------------------------------------------------------------------------------
# PATCH OVERLAP FUNCTIONS
#------------------------------------------------------------------------------

def minimal_cost_path(B1, B2, block_dim, overlap):
    # Assume looking at column overlap
    B1_ov = convert_to_grayscale(B1[:, (block_dim[1] - overlap):, :])
    B2_ov = convert_to_grayscale(B2[:, :overlap, :])

    # Dynamic Programming for cut
    error_surface = np.power(B1_ov - B2_ov, 2)

    dp = np.zeros(error_surface.shape)
    paths = {}
    for i in range(dp.shape[0]):
        for j in range(dp.shape[1]):
            dp[i, j] = error_surface[i, j]

            if i == 0:
                paths[(i, j)] = [(i, j)]
            else:
                min_list = [(dp[i - 1, j], j)]
                if j - 1 >= 0:
                    min_list.append((dp[i - 1, j - 1], j - 1))
                if j + 1 < dp.shape[1]:
                    min_list.append((dp[i - 1, j + 1], j + 1))

                prev_min = min(min_list)
                dp[i, j] += prev_min[0]

                paths[(i, j)] = copy.copy(paths[i - 1, prev_min[1]])
                paths[(i, j)].append((i, j))

    # Find min cost path
    min_index = 0
    for i in range(dp.shape[1]):
        if dp[-1, i] < dp[-1, min_index]:
            min_index = i

    min_cost_path = paths[(dp.shape[0] - 1, min_index)]

    return min_cost_path

def get_row_overlap_regions(neighbor, blocks, block_dim, overlap):
    # Get row overlap region from already placed neighbor
    neighbor_row_region = neighbor[(neighbor.shape[0] - overlap):, :, :]

    # Get row overlap regions of candidate texture blocks
    block_row_regions = blocks[:, :, :overlap, :, :]

    return neighbor_row_region, block_row_regions

def get_col_overlap_regions(neighbor, blocks, block_dim, overlap):
    # Get col overlap region from already placed neighbor
    neighbor_col_region = neighbor[:, (neighbor.shape[1] - overlap):, :]

    # Get col overlap regions of candidate texture blocks
    block_col_regions = blocks[:, :, :, :overlap, :]

    return neighbor_col_region, block_col_regions

#------------------------------------------------------------------------------
# BLOCK ERROR FUNCTIONS
#------------------------------------------------------------------------------

def get_valid_blocks(neighbor_overlap, block_overlaps, scale=0.1):
    # Get matrix of L2 norms between block overlaps and neighbor overlap
    error_matrix = get_error_matrix(neighbor_overlap, block_overlaps)

    # Find error tolerance
    t = get_error_t(error_matrix, scale)

    return error_matrix <= t

def get_error_matrix(neighbor_overlap, block_overlaps):
    # Calculate L2 norm between overlaps
    error_matrix = np.power(block_overlaps - neighbor_overlap, 2)
    while len(error_matrix.shape) > 2:
        error_matrix = np.sum(error_matrix, axis=2)

    error_matrix = np.sqrt(error_matrix)

    # For 2D case
    # error_matrix = np.linalg.norm(block_overlaps - neighbor_overlap, axis=(2, 3))

    return error_matrix

def get_error_t(error_matrix, scale=0.1):
    num_vals = error_matrix.shape[0]*error_matrix.shape[1]

    # Grab value of second smallest total error and use to define error tolerance
    min_block_error = np.partition(np.reshape(error_matrix, num_vals), 1)[1]

    t = min_block_error + min_block_error * scale

    return t

#------------------------------------------------------------------------------
# IMAGE IO FUNCTIONS
#------------------------------------------------------------------------------

def read_image(path: Union[Path, str]) -> np.ndarray:
    '''
    Read a PNG or JPG image an array of linear RGB radiance values ∈ [0,1].
    '''
    return (np.float32(Image.open(path)) / 255)**2.2


def write_image(path: Union[Path, str], image: np.ndarray) -> None:
    '''
    Write an array of linear RGB radiance values ∈ [0,1] as a PNG or JPG image.
    '''
    Image.fromarray(np.uint8(255 * image.clip(0, 1)**(1/2.2))).save(path)

#------------------------------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS
#------------------------------------------------------------------------------

def convert_to_grayscale(im):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    Y = 0.3 * R + 0.6 * G + 0.1 * B
    return Y
