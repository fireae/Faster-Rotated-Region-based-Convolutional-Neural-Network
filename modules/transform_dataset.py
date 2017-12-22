class TransformDataset(object):

    """Dataset that indexes data of a base dataset and transforms it.

    This dataset wraps a base dataset by modifying the behavior of the base
    dataset's :meth:`__getitem__`. Arrays returned by :meth:`__getitem__` of
    the base dataset with an integer index are transformed by the given
    function :obj:`transform`.


    Args:
        dataset: Underlying dataset. The index of this dataset corresponds
            to the index of the base dataset.
        transform (callable): A function that is called to transform values
            returned by the underlying dataset's :meth:`__getitem__`.

    """

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        in_data = self._dataset[index]
        if isinstance(index, slice):
            return [self._transform(in_data_elem) for in_data_elem in in_data]
        else:
            return self._transform(in_data)

    def __len__(self):
        return len(self._dataset)
