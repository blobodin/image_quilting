import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pathlib import Path
from typing import Tuple, Union

from PIL import Image

def combine_cuts(h_cut, v_cut):
    v_cut_dict = dict([(v_cut[i], i) for i in range(len(v_cut))])
    h_cut_dict = dict([(h_cut[i], i) for i in range(len(h_cut))])

    for m in range(len(h_cut) - 1, -1, -1):
        (i, j) = h_cut[m]

        if (i, j) in v_cut_dict:
            n = v_cut_dict[(i, j)]
            return h_cut[m:], v_cut[(n + 1):]
        elif (i + 1, j) in v_cut_dict:
            n = v_cut_dict[(i + 1, j)]
            return h_cut[m:], v_cut[n:]
        elif (i + 1, j - 1) in v_cut_dict:
            n = v_cut_dict[(i + 1, j - 1)]
            return h_cut[m:], v_cut[n:]
        elif (i, j - 1) in v_cut_dict:
            n = v_cut_dict[(i, j - 1)]
            return h_cut[m:], v_cut[n:]
        elif (i - 1, j - 1) in v_cut_dict:
            n = v_cut_dict[(i - 1, j - 1)]
            return h_cut[m:], v_cut[n:]

    print("Shouldnt be here")
    assert False

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
    for i in range(len(paths)):
        if dp[-1, i] < dp[-1, min_index]:
            min_index = i

    min_cost_path = paths[min_index]
    min_cost_path.append((dp.shape[0] - 1, min_index))

    return min_cost_path

def calc_error(B1, B2):
    return np.linalg.norm(B1 - B2)

def get_row_overlap_regions(neighbor, blocks, block_dim, overlap):
    # Get row overlap region from already placed neighbor
    neighbor_row_region = neighbor[(block_dim[0] - overlap):, :]

    # Get row overlap regions of candidate texture blocks
    block_row_regions = blocks[:, :, :overlap, :]

    # Insert dim if needed!

    return neighbor_row_region, block_row_regions

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
    # return error_matrix < error_tol

    min_block_error = np.partition(np.reshape(error_matrix, (error_matrix.shape[0]**2)), 1)[1]
    print(min_block_error)
    print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    t = min_block_error + min_block_error
    print(t)
    print(error_matrix <= t)
    return error_matrix <= t

def get_error_matrix(neighbor_overlap, block_overlaps, error_tol):
    # Calculate L2 norm between overlaps
    error_matrix = np.linalg.norm(block_overlaps - neighbor_overlap, axis=(2, 3))

    # Get boolean matrix of blocks whose error overlap satifies the error tolerance
    # return error_matrix < error_tol

    return error_matrix

def find_valid_block(row_neighbor, col_neighbor, texture, block_dim, overlap, error_tol):
    # Get all candidate inserts from base texture
    blocks = sliding_window_view(texture, block_dim)

    # Prepare boolean matrix that will designate blocks that satisfy error
    # constraints in row + col overlap regions
    valid_blocks = np.ones((blocks.shape[0], blocks.shape[1]))
    error_matrix = np.zeros((blocks.shape[0], blocks.shape[1]))
    flag = False
    # If testing error tolerance horizontally
    if row_neighbor is not None:
        blocks = blocks[:, :, :, :row_neighbor.shape[1]]

        # Get row overlap regions of neighbor + blocks
        neighbor_row_region, block_row_regions = get_row_overlap_regions(row_neighbor, blocks, block_dim, overlap)

        # Get boolean matrix of blocks that satisfy error tolerance in row
        # overlap region
        # row_valid_blocks = get_valid_blocks(neighbor_row_region, block_row_regions, error_tol)
        #
        # print(np.sum(row_valid_blocks))
        #
        # # Adjust overall boolean matrix with new info
        # valid_blocks *= row_valid_blocks

        row_error_matrix = get_error_matrix(neighbor_row_region, block_row_regions, error_tol)
        error_matrix += row_error_matrix
        flag = True


    # If testing error tolerance vertically
    if col_neighbor is not None:
        blocks = blocks[:, :, :col_neighbor.shape[0], :]

        # Get col overlap regions of neighbor + blocks
        neighbor_col_region, block_col_regions = get_col_overlap_regions(col_neighbor, blocks, block_dim, overlap)

        # Get boolean matrix of blocks that satisfy error tolerance in col
        # overlap region
        # col_valid_blocks = get_valid_blocks(neighbor_col_region, block_col_regions, error_tol)
        #
        # print(np.sum(col_valid_blocks))
        #
        # # Adjust overall boolean matrix with new info
        # valid_blocks *= col_valid_blocks

        col_error_matrix = get_error_matrix(neighbor_col_region, block_col_regions, error_tol)
        error_matrix += col_error_matrix
        flag = True

    if not flag:
        error_matrix = valid_blocks

    min_block_error = np.partition(np.reshape(error_matrix, (error_matrix.shape[0]**2)), 1)[1]
    t = min_block_error + min_block_error * .1
    valid_blocks = error_matrix <= t

    # Get all indexes of blocks whose overlap satify the error tolerance
    valid_block_indexes = np.argwhere(valid_blocks)

    # Randomly choose index of valid block
    rnd_choice = np.random.choice(range(len(valid_block_indexes)))
    chosen_block_index = valid_block_indexes[rnd_choice]

    # Grab randomly chosen block
    chosen_block = blocks[chosen_block_index[0], chosen_block_index[1], :, :]

    return np.copy(chosen_block)

def insert_block(quilt, x, y, block, block_dim, h_cut, v_cut):
    wipe = np.zeros(block_dim)

    if h_cut is not None:
        for cut_loc in h_cut:
            (i, j) = cut_loc

            wipe[:i, j] = 1
            block[:i, j] = 0

    if v_cut is not None:
        for cut_loc in v_cut:
            (i, j) = cut_loc

            wipe[i, :j] = 1
            block[i, :j] = 0

    if h_cut is not None and v_cut is not None:
        # Check if first cut loc for h_cut lies on top of first cut_loc for
        # v_cut, if it does use it to clear out remaining area. Otherwise
        # use first v_cut
        if h_cut[0][0] == v_cut[0][0] - 1 and h_cut[0][1] == v_cut[0][1]:
            wipe[:(h_cut[0][0] + 1), :(h_cut[0][1])] = 1
            block[:(h_cut[0][0] + 1), :(h_cut[0][1])] = 0
        else:
            wipe[:(v_cut[0][0]), :(v_cut[0][1] + 1)] = 1
            block[:(v_cut[0][0]), :(v_cut[0][1] + 1)] = 0

    # Cut wipe and block if at right edge or bottom boundary before inserting
    quilt[x:(x + block_dim[0]), y:(y + block_dim[1])] *= wipe[:min(quilt.shape[0] - x, block_dim[0]), :min(quilt.shape[1] - y, block_dim[1])]
    quilt[x:(x + block_dim[0]), y:(y + block_dim[1])] += block[:min(quilt.shape[0] - x, block_dim[0]), :min(quilt.shape[1] - y, block_dim[1])]

    return quilt

def make_quilt(texture, output_dim, block_dim, overlap, error_tol):

    output = np.zeros(output_dim)

    row_step = block_dim[0] - overlap
    col_step = block_dim[1] - overlap
    for i in range(0, output_dim[0], row_step):
        for j in range(0, output_dim[1], col_step):
            row_neighbor = None
            col_neighbor = None

            # print(i, j, col_step, output_dim)
            if i != 0:
                row_neighbor = output[(i - row_step):(i - row_step + block_dim[0]), j:(j + block_dim[1])]
            if j != 0:
                col_neighbor = output[i:(i + block_dim[0]), (j - col_step):(j - col_step + block_dim[1])]

            # Get Valid Block to Insert
            new_block = find_valid_block(row_neighbor, col_neighbor, texture, block_dim, overlap, error_tol)

            h_cut = None
            v_cut = None
            # Compute Min Cost Path
            if row_neighbor is not None:
                h_cut = [(r, q) for (q, r) in minimal_cost_path(row_neighbor.T, new_block.T, block_dim, overlap)]
            if col_neighbor is not None:
                v_cut = minimal_cost_path(col_neighbor, new_block, block_dim, overlap)

            if h_cut is not None or v_cut is not None:
                if h_cut is not None and v_cut is not None:
                    h_cut, v_cut = combine_cuts(h_cut, v_cut)

                # Insert Block
                output = insert_block(output, i, j, new_block, block_dim, h_cut, v_cut)
            else:
                output[:block_dim[0], :block_dim[1]] = new_block

        write_image("new_brick.png", output)


    return output

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

def convert_to_grayscale(im):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    Y = 0.3 * R + 0.6 * G + 0.1 * B
    return Y

if __name__ == "__main__":
    # Load Texture
    im = read_image("brick.jpg")
    write_image("new_brick.png", im)
    gray_im = convert_to_grayscale(im)

    # Problems with block clipping
    block_dim = (20, 20)
    overlap = int(block_dim[0] / 6)
    error_tol = 2.2
    output_dim = (450, 450)

    new_im = make_quilt(gray_im, output_dim, block_dim, overlap, error_tol)

    # Save Quilt
    write_image("new_brick.png", new_im)
