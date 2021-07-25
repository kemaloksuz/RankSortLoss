from functools import partial

import torch
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret
    
def vectorize_labels(flat_labels, num_classes, label_weights = None):
    prediction_number = flat_labels.shape[0]
    labels = torch.zeros( [prediction_number, num_classes], device=flat_labels.device)
    pos_labels = flat_labels < num_classes
    labels[pos_labels, flat_labels[pos_labels]] = 1
    if label_weights is not None:
        ignore_labels = (label_weights == 0)
        labels[ignore_labels, :] = -1
    return labels.reshape(-1)
