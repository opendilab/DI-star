import torch


def l2_distance(a, b, min=0, max=0.8, threshold=5, spatial_x=160):
    x0 = a % spatial_x
    y0 = a // spatial_x
    x1 = b % spatial_x
    y1 = b // spatial_x
    l2 = torch.sqrt((torch.square(x1 - x0) + torch.square(y1 - y0)).float())
    cost = (l2 / threshold).clamp_(min=min, max=max)
    return cost


def levenshtein_distance(behaviour, target, behaviour_extra=None, target_extra=None, extra_fn=None):
    r"""
    Overview:
        Levenshtein Distance(Edit Distance)

    Arguments:
        Note:
            N1 >= 0, N2 >= 0

        - behaviour (:obj:`torch.LongTensor`): shape[N1]
        - target (:obj:`torch.LongTensor`): shape[N2]
        - behaviour_extra (:obj:`torch.Tensor or None`)
        - target_extra (:obj:`torch.Tensor or None`)
        - extra_fn (:obj:`function or None`): if specified, the distance metric of the extra input data

    Returns:
        - (:obj:`torch.FloatTensor`) distance(scalar), shape[1]

    Test:
        torch_utils/network/tests/test_metric.py
    """
    assert (isinstance(behaviour, torch.Tensor) and isinstance(target, torch.Tensor))
    assert behaviour.dtype == target.dtype, f'bahaviour_dtype: {behaviour.dtype}, target_dtype: {target.dtype}'
    assert (behaviour.device == target.device)
    assert (type(behaviour_extra) == type(target_extra))
    if not extra_fn:
        assert (not behaviour_extra)
    N1, N2 = behaviour.shape[0], target.shape[0]
    assert (N1 >= 0 and N2 >= 0)
    if N1 == 0 or N2 == 0:
        distance = max(N1, N2)
    else:
        dp_array = torch.zeros(N1 + 1, N2 + 1).float()
        dp_array[0, :] = torch.arange(0, N2 + 1)
        dp_array[:, 0] = torch.arange(0, N1 + 1)
        for i in range(1, N1 + 1):
            for j in range(1, N2 + 1):
                if behaviour[i - 1] == target[j - 1]:
                    if extra_fn:
                        dp_array[i, j] = dp_array[i - 1, j - 1] + extra_fn(behaviour_extra[i - 1], target_extra[j - 1])
                    else:
                        dp_array[i, j] = dp_array[i - 1, j - 1]
                else:
                    dp_array[i, j] = min(dp_array[i - 1, j] + 1, dp_array[i, j - 1] + 1, dp_array[i - 1, j - 1] + 1)
        distance = dp_array[N1, N2]
    return torch.as_tensor(distance).to(behaviour.device)


def hamming_distance(behaviour, target):
    r'''
    Overview:
        Hamming Distance

    Arguments:
        Note:
            behaviour, target are also boolean vector(0 or 1)

        - behaviour (:obj:`torch.LongTensor`): behaviour input, shape[B, N], while B is the batch size
        - target (:obj:`torch.LongTensor`): target input, shape[B, N], while B is the batch size

    Returns:
        - distance(:obj:`torch.LongTensor`): distance(scalar), the shape[1]

    Shapes:
        - behaviour & target (:obj:`torch.LongTensor`): shape :math:`(B, N)`, \
            while B is the batch size and N is the dimension

    Test:
        torch_utils/network/tests/test_metric.py
    '''
    assert (isinstance(behaviour, torch.Tensor) and isinstance(target, torch.Tensor))
    assert behaviour.dtype == target.dtype, f'bahaviour_dtype: {behaviour.dtype}, target_dtype: {target.dtype}'
    assert (behaviour.device == target.device)
    assert (behaviour.shape == target.shape)
    return behaviour.ne(target).sum(dim=-1).float()

