import numpy as np
from utils.quilt_lib import *

def make_quilt(texture, output_dim, block_dim, overlap):
    # Prepare output quilt matrix
    output = np.zeros(output_dim)

    # Compute step size for iterations
    row_step = block_dim[0] - overlap
    col_step = block_dim[1] - overlap

    blocks = get_block_set(texture, block_dim)

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

            # Get Valid Block to Insert
            new_block = find_valid_quilt_block(row_neighbor, col_neighbor, blocks, block_dim, overlap)

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

    return output

if __name__ == "__main__":
    textures = ["text"]

    for texture in textures:
        # Load Texture
        im = read_image(f"textures/{texture}.jpg")
        if len(im.shape) == 2:
            new_im = np.zeros((im.shape[0], im.shape[1], 3))
            new_im[:, :, 0] = im
            new_im[:, :, 1] = im
            new_im[:, :, 2] = im
            im = new_im

        output_dim = (500, 500, 3)
        block_dim = (60, 60, 3)
        overlap = int(block_dim[0] / 6)
        new_im = make_quilt(im, output_dim, block_dim, overlap)

        # Save Quilt
        write_image(f"generated_textures/new_{texture}.png", new_im)
