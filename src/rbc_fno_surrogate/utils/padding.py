import torch


def pad(x, dim, extent, padding_mode):
    fn = {"circular": pad_circular, "replicate": pad_replicate, "zeros": pad_zeros}[
        padding_mode
    ]
    return fn(x, dim, extent)


def slices_from_to(shape, dim, from_idx, to_idx):
    return (
        [slice(0, shape[k], 1) for k in range(dim)]
        + [slice(from_idx, to_idx, 1)]
        + [slice(0, shape[k], 1) for k in range(dim + 1, len(shape))]
    )


def raw_extend_tensor(x, dim, extent):
    shape = [*x.shape[:dim], x.shape[dim] + 2 * extent, *x.shape[dim + 1 :]]
    slices = slices_from_to(shape, dim, extent, extent + x.shape[dim])
    y = x.new_empty(shape)
    y[slices] = x
    return y


def pad_zeros(x, dim, extent):
    y = raw_extend_tensor(x, dim, extent)
    index = torch.tensor(
        [i for i in range(extent)] + [y.shape[dim] - i - 1 for i in range(extent)],
        device=x.device,
    )
    y.index_fill_(dim, index, 0)
    return y


def pad_replicate(x, dim, extent):
    y = raw_extend_tensor(x, dim, extent)
    slices_btm_y = slices_from_to(y.shape, dim, 0, extent)
    slices_btm_x = slices_from_to(x.shape, dim, 0, extent)
    y[slices_btm_y] = torch.flip(x[slices_btm_x], (dim,))
    slices_top_y = slices_from_to(y.shape, dim, y.shape[dim] - extent, y.shape[dim])
    slices_top_x = slices_from_to(x.shape, dim, x.shape[dim] - extent, x.shape[dim])
    y[slices_top_y] = torch.flip(x[slices_top_x], (dim,))
    return y


def pad_circular(x, dim, extent):
    y = raw_extend_tensor(x, dim, extent)
    slices_btm_y = slices_from_to(y.shape, dim, 0, extent)
    slices_btm_x = slices_from_to(x.shape, dim, x.shape[dim] - extent, x.shape[dim])
    y[slices_btm_y] = x[slices_btm_x]
    slices_top_y = slices_from_to(y.shape, dim, y.shape[dim] - extent, y.shape[dim])
    slices_top_x = slices_from_to(x.shape, dim, 0, extent)
    y[slices_top_y] = x[slices_top_x]
    return y
