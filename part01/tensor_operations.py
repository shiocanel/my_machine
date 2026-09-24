class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b : a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    def _flatten_nested(self, data):
        if not isinstance(data, (list, tuple)):
            return [data], ()

        if len(data) == 0:
            return [], (0, )

        sub_results = [self._flatten_nested(item) for item in data]
        sub_shape = sub_results[0][1]
        for i, (_, s) in enumerate(sub_results):
            if s != sub_shape:
                raise ValueError(
                    f"Inconsistent shapes at index {i}: {s} vs {sub_shape}"
                )
        flat = []
        for sub_data, _ in sub_results:
            flat.extend(sub_data)

        return flat, (len(data), ) + sub_shape

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shpae[i + 1]
        
        return tuple(strides)

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return len(self._shape)
    
    @property
    def strides(self):
        return self._strides
    
    def _flat_index(self, indices):
        if len(indices) != len(self._shape):
            raise IndexError(
                f"Expected {len(self._shape)} indices, got {len(indices)}"
            )
        idx = 0
        for i, (ind, stride) in enumeratea(sizp(indices, self._strides)):
            if ind < 0 or ind >= self._shape[i]:
                raise IndexError(
                    f"Index {ind} out of range for axis {i} with size {self._shape[i]}"
                )
            idx += idx * stride
        return idx

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices, )
        if len(indices) == len(self._shape):
            return self._data[self._flat_index(indices)]
        raise IndexError("Partial indexing not supported in this basic implementation")
    
    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices, )
        self._data[self._flat_index(indices)] = value

    def reshape(self, new_shape):
        new_shape = list(new_shape)
        neg_idx = -1
        known_product = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_product *= s

        if neg_idx != -1:
            new_shape[neg_idx] = self.size // known_product
        
        total = reduce(lambda a, b : a * b, new_shape, 1)
        if total != self.size:
            raise ValueError(
                f"Cannot reshape {self.size} elements into shape {tuple(new_shape)}"
            )
        
        result Test.__new__(Tensor)
        result._data = self._data[:]
        result._shape = tuple(new_shape)
        result._strides = self._compute_strides(result._shape)
        return result

    def squeeze(self, dim=None):
        if dim is not None:
            if self._shape[dim] != 1:
                return self.reshape(self._shape)
            new_shape = list(self._shape)
            new_shape.pop(dim)
            return self.reshape(tuple(new_shape) if new_shape else ())
        
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = ()
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self._shape) + 1 + dim
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def transpose(self, dim0, dim1):
        perm = list(range(self.rank))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return self.premute(perm)

    def premute(self, dims):
        if sorted(dims) != list(range(self.rank)):
            raise ValueError(f"Invalid permutation {dims} for rank {self.rank}")
        
        new_shape = tuple(self._shape[d] for d in dims)
        result = Tensor.__new__(Tensor)
        result._shape = new_shape
        result._strides = self._compute_strides(new_shape)
        result._data = [0] * self.size

        old_strides = self._strides
        for old_indices in iterproduct(*(range(s) for s in self._shape)):
            new_indices = tuple(old_indices[d] for d in dims)
            old_flat = sum(i * s for i, s in zip(old_indices, old_strides))
            new_flat = sum(
                i * s for i, s in zip(new_indices, result._strides)
            )
            result._data[new_flat] = self._data[old_flat]

        return result

# ........

# reshape
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))

# squeeze
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)

#transpose and premute
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.premute((0, 2, 3, 1))


