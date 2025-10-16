import numpy as np
class Sp_tensor:
    def __init__(self, coords, values, tensor_size, allow_duplication=False, normalize=False, check_empty=False, sort=True):
        
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords, dtype=np.int64)
            
        assert len(coords) == len(values), "#coord and #values need to be same"
        if sort:
            sort_keys = tuple( coords[:, col] for col in range(coords.shape[1]-1,-1,-1))
            coords = coords[ np.lexsort(sort_keys) ]
            values = values[ np.lexsort(sort_keys) ]
        assert len(coords) == len(values), "#coord and #values need to be same"
        
        self.coords = np.array(coords, dtype=np.int64)
        self.values = np.array(values, dtype=np.float64)

        if not(allow_duplication):
            self.coords, self.values = self.kill_duplication()
        
        self.tensor_size = tensor_size
        self.nnz = len(self.values)
        
        if isinstance(tensor_size, int):
            self.tensor_dim  = 1
        else:
            self.tensor_dim  = len(tensor_size)
            
        if normalize:
            self.normalize()
            
        if check_empty:
            self.see_empty()
        
        if self.tensor_dim == 1:
            self.coord_to_value = { self.coords[n] : self.values[n] for n in range(self.nnz) }
        else:
            self.coord_to_value = { tuple(self.coords[n]) : self.values[n] for n in range(self.nnz) }

    def update_coord_to_value(self):
        if self.tensor_dim == 1:
            self.coord_to_value = { self.coords[n] : self.values[n] for n in range(self.nnz) }
        else:
            self.coord_to_value = { tuple(self.coords[n]) : self.values[n] for n in range(self.nnz) }
    
    def see_empty(self):
        for d in range(self.tensor_dim):
            assert len(np.unique(self.coords[:,d])) == self.tensor_size[d], "empty label exists"
            
    def normalize(self):
        self.values /= np.sum(self.values)

    def kill_duplication(self):
        coords_tuple = [tuple(row) for row in self.coords]
        coords_array = np.array(coords_tuple)
        unique_coords, inverse_indices = np.unique(self.coords, axis=0, return_inverse=True)
    
        values_uniq = np.zeros(len(unique_coords))
        for i, idx in enumerate(inverse_indices):
            values_uniq[idx] += self.values[i]

        return unique_coords, values_uniq


def dense_to_sparse(X):
    tensor_size = np.shape(X)
    coords = np.array([list(index) for index in np.ndindex(X.shape)])
    spt = Sp_tensor(coords, X.reshape(-1), X.shape)
    
    return spt
