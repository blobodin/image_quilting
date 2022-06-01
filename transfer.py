import numpy as np
from utils.quilt_lib import *

def make_transfer(output, texture, c_texture, c_target, block_dim, overlap, alpha, N):
    output_dim = output.shape

    # Compute step size for iterations
    row_step = block_dim[0] - overlap
    col_step = block_dim[1] - overlap

    blocks = get_block_set(texture, block_dim)

    c_blocks = sliding_window_view(c_texture, (block_dim[0], block_dim[1]))

    for i in range(0, output_dim[0], row_step):
        for j in range(0, output_dim[1], col_step):
            row_neighbor = None
            col_neighbor = None

            # Grab row and col neighbors if they exist
            if i != 0:
                prev_i = i - row_step
                row_neighbor = output[prev_i:(prev_i + block_dim[0]), j:(j + block_dim[1]), :]
            if j != 0:
                prev_j = j - col_step
                col_neighbor = output[i:(i + block_dim[0]), prev_j:(prev_j + block_dim[1]), :]

            if N == 0:
                curr_region = None
            else:
                curr_region = output[i:(i + block_dim[0]), j:(j + block_dim[1])]

            curr_c_target = c_target[i:(i + block_dim[0]), j:(j + block_dim[1])]

            # Get Valid Block to Insert
            new_block = find_valid_transfer_block(row_neighbor, col_neighbor, blocks, block_dim, overlap, curr_region, c_blocks, curr_c_target, alpha)

            # Chop block to fit into output
            new_block = new_block[:min(output.shape[0] - i, block_dim[0]), :min(output.shape[1] - j, block_dim[1]), :]

            h_cut = None
            v_cut = None
            # Compute  horizontal and vertical cuts
            if row_neighbor is not None:
                # Flip images to be vertical for use with algorithm
                flip_neighbor = np.transpose(row_neighbor, (1, 0, 2))
                flip_block = np.transpose(new_block, (1, 0, 2))

                # Flip indexes returned for correct coordinates
                h_cut = [(r, q) for (q, r) in minimal_cost_path(flip_neighbor, flip_block, block_dim, overlap)]
            if col_neighbor is not None:
                v_cut = minimal_cost_path(col_neighbor, new_block, block_dim, overlap)

            # Insert block while applying cut(s)
            if h_cut is not None or v_cut is not None:
                # Insert Block
                output = insert_block(output, i, j, new_block, block_dim, h_cut, v_cut)
            else:
                # If first block, just insert
                output[:block_dim[0], :block_dim[1], :] = new_block

            write_image("transfer.png", output)
    return output


if __name__ == "__main__":
    texture_name = "orange"
    target_name = "pear"

    # Load correspondence map of target
    c_target = convert_to_grayscale(read_image(f"targets/{target_name}.jpg"))

    # Load Texture and correspondence map of texture
    texture = read_image(f"textures/{texture_name}.jpg")
    c_texture = convert_to_grayscale(texture)

    # Fix 2D source textures to be RGB
    if len(texture.shape) == 2:
        new_im = np.zeros((texture.shape[0], texture.shape[1], 3))
        new_im[:, :, 0] = texture
        new_im[:, :, 1] = texture
        new_im[:, :, 2] = texture
        texture = new_im

    # Define initial params
    output_dim = (c_target.shape[0], c_target.shape[1], 3)
    output = np.zeros(output_dim)
    N = 3
    block_dim = (30, 30, 3)

    # Apply quilting multiple times with decreasing block size
    # and increasing alpha
    for n in range(N):
        alpha = 0.8 * (n / (N - 1)) + 0.1
        overlap = int(block_dim[0] / 5)

        output = make_transfer(output, texture, c_texture, c_target, block_dim, overlap, alpha, N)

        block_dim = (int(block_dim[0] -10), int(block_dim[1] -10), 3)

    # Save Quilt
    write_image(f"transfers/new_transfer_{texture_name}_{target_name}.png", output)
