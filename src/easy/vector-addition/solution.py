# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Convert raw pointers to typed pointers for float32
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    # Get the program ID
    pid = tl.program_id(axis=0)

    # Calculate the block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask for valid elements
    mask = offsets < n_elements

    # Load data from input vectors
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Perform the addition
    c = a + b

    # Store the result
    tl.store(c_ptr + offsets, c, mask=mask)


# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE)
