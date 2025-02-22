"""
Saksham's Custom Matrix Implementation (Goel's Python)

Author: Saksham Goel
Date: Feb 15, 2025
Version: 3.0

Github: @SakshamG7
Organization: AceIQ
Website: https://aceiq.ca
Contact: mail@aceiq.ca
Location: 
"""

from __future__ import annotations # Required for type hinting the class itself

### matrix class
# rows: the number of rows in the matrix
# cols: the number of columns in the matrix
# data: the data of the matrix
class matrix(object):
    # __init__: the constructor for the matrix class
    def __init__(self, data: list = [], rows: int = -1, cols: int = -1) -> None:
        self.rows = rows
        self.cols = cols
        self.data = data
        if len(data) == 0 and rows == -1 and cols == -1:
            raise ValueError("No data or dimensions provided")

        # If data is only 1D, convert it to 2D
        if len(data) != 0:
            if type(data[0]) != list:
                self.data = [data]

        if len(data) != 0:
            self.rows = len(self.data)
            self.cols = len(self.data[0])
        if rows != -1 and cols != -1:
            self.data = [[0 for i in range(cols)] for j in range(rows)]
    
    # transpose: transposes the matrix itself
    def transpose(self) -> None:
        new_data = [[0 for i in range(self.rows)] for j in range(self.cols)]
        for i in range(self.rows):
            for j in range(self.cols):
                new_data[j][i] = self.data[i][j]
        
        self.data = new_data
        self.rows, self.cols = self.cols, self.rows
    
    # apply: applies a function to each element of the matrix
    # func: the function to apply
    def apply(self, func) -> matrix:
        result = matrix(self.data.copy())
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(result.data[i][j])
        return result

    # copy: returns a copy of the matrix
    def copy(self) -> matrix:
        return matrix(self.data.copy())
    
    # __add__: adds two matrices together
    def __add__(self, other: matrix) -> matrix:
        if self.rows != other.rows or self.cols != other.cols:
            if self.rows == other.cols and self.cols == other.rows:
                other.transpose()
            else:
                return None
        result = matrix(self.data.copy())
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] += other.data[i][j]
        return result
    
    # __sub__: subtracts two matrices
    def __sub__(self, other: matrix) -> matrix:
        if self.rows != other.rows or self.cols != other.cols:
            return None
        result = matrix(self.data.copy())
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] -= other.data[i][j]
        return result
    
    # __mul__: multiplies two matrices, returns the dot product
    def __mul__(self, other: matrix) -> matrix:
        return dot_product(self, other)
    
    # __str__: returns the string representation of the matrix
    def __str__(self) -> str:
        result = ""
        for i in range(self.rows):
            for j in range(self.cols):
                result += str(self.data[i][j]) + " "
            result += "\n"
        return result
    
    # __len__: returns a list of the dimensions of the matrix
    def __len__(self) -> list:
        return [self.rows, self.cols]
    
    ### append_row, appends a row to the matrix
    # row: the row to append
    def append_row(self, row: list) -> None:
        if len(row) != self.cols:
            return
        self.data.append(row)
        self.rows += 1
    
    ### append_col, appends a column to the matrix
    # col: the column to append
    def append_col(self, col: list) -> None:
        if len(col) != self.rows:
            return
        for i in range(self.rows):
            self.data[i].append(col[i])
        self.cols += 1
    
    ### remove_row, removes a row from the matrix
    # index: the index of the row to remove
    def remove_row(self, index: int) -> None:
        if index >= self.rows:
            return
        self.data.pop(index)
        self.rows -= 1

    ### remove_col, removes a column from the matrix
    # index: the index of the column to remove
    def remove_col(self, index: int) -> None:
        if index >= self.cols:
            return
        for i in range(self.rows):
            self.data[i].pop(index)
        self.cols -= 1
    
    ### sum, returns the sum of all elements in the matrix
    def sum(self) -> float:
        result = 0
        for i in range(self.rows):
            for j in range(self.cols):
                result += self.data[i][j]
        return result


### dot_product, returns the dot product of two matrices, auto transposing built in
# mat1: the first matrix
# mat2: the second matrix
def dot_product(mat1: matrix, mat2: matrix) -> matrix:
    if mat1.cols != mat2.rows:
        if mat1.cols == mat2.cols and mat1.rows == mat2.rows: # If the matrices are the same size, transpose one
            mat2.transpose()
        else:
            return None
    result = matrix(rows=mat1.rows, cols=mat2.cols)
    for i in range(mat1.rows):
        for j in range(mat2.cols):
            for k in range(mat1.cols):
                result.data[i][j] += mat1.data[i][k] * mat2.data[k][j]
    return result