# -*- coding: utf-8 -*-
#
# Copyright 2023, 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np

cimport cython
cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cnp.import_array()
DTYPE = np.int32
ctypedef cnp.int32_t DTYPE_t


# Format specification::
#
# Big endian bytes order
# 8 bit - 0x02 (current format specification)
# 12 bit - number of atoms
# 12 bit - cis/trans stereo block size
#
# Atom block 9 bytes (repeated):
# 12 bit - atom number
# 4 bit - number of neighbors
# 2 bit tetrahedron sign (00 - not stereo, 10 or 11 - has stereo)
# 2 bit - allene sign
# 5 bit - isotope (00000 - not specified, over = isotope - common_isotope + 16)
# 7 bit - atomic number (<=118)
# 32 bit - XY float16 coordinates
# 3 bit - hydrogens (0-7). Note: 7 == None
# 4 bit - charge (charge + 4. possible range -4 - 4)
# 1 bit - radical state
# Connection table: flatten list of neighbors. neighbors count stored in atom block.
# For example CC(=O)O - {1: [2], 2: [1, 3, 4], 3: [2], 4: [2]} >> [2, 1, 3, 4, 2, 2].
# Repeated block (equal to bonds count).
# 24 bit - paired 12 bit numbers.
# Bonds order block 3 bit per bond zero-padded to full byte at the end.
# Cis/trans data block (repeated):
# 24 bit - atoms pair
# 7 bit - zero padding. in future can be used for extra bond-level stereo, like atropoisomers.
# 1 bit - sign

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def unpack(const unsigned char[::1] data not None, unsigned short add_cls, unsigned short symmetric_attention,
           unsigned short components_attention, DTYPE_t max_neighbors, DTYPE_t max_distance):
    """
    Optimized chython pack to graph tensor converter.
    Ignores charge, radicals, isotope, coordinates, bond order, and stereo info
    """
    cdef unsigned char a, b, c, hydrogens, neighbors_count
    cdef unsigned char *connections

    cdef unsigned short atoms_count, bonds_count = 0, order_count = 0, cis_trans_count
    cdef unsigned short i, j, k, n, m
    cdef unsigned short[4096] mapping
    cdef unsigned int size, shift = 4

    cdef cnp.ndarray[DTYPE_t, ndim=1] atoms, neighbors
    cdef cnp.ndarray[DTYPE_t, ndim=2] distance
    cdef DTYPE_t d

    # read header
    if data[0] != 2:
        raise ValueError('invalid pack version')

    a, b, c = data[1], data[2], data[3]
    atoms_count = (a << 4| b >> 4) + add_cls
    cis_trans_count = (b & 0x0f) << 8 | c

    atoms = np.empty(atoms_count, dtype=DTYPE)
    neighbors = np.zeros(atoms_count, dtype=DTYPE)
    distance = np.full((atoms_count, atoms_count), 9999, dtype=DTYPE)  # fill with unreachable value

    # allocate memory
    connections = <unsigned char*> PyMem_Malloc(atoms_count * sizeof(unsigned char))
    if not connections:
        raise MemoryError()

    if add_cls:
        atoms[0] = 1
        neighbors[0] = 0
        distance[0] = 1  # set CLS to all atoms attention

        if symmetric_attention:  # set all atoms to CLS attention
            distance[1:, 0] = 1
        else:  # disable atom to CLS attention
            distance[1:, 0] = 0

    # unpack atom block
    for i in range(add_cls, atoms_count):
        distance[i, i] = 0  # set diagonal to zero
        a, b = data[shift], data[shift + 1]
        n = a << 4 | b >> 4
        mapping[n] = i
        connections[i] = neighbors_count = b & 0x0f
        bonds_count += neighbors_count

        atoms[i] = (data[shift + 3] & 0x7f) + 2

        hydrogens = data[shift + 8] >> 5
        if hydrogens != 7:  # hydrogens is not None
            neighbors_count += hydrogens
        if neighbors_count > max_neighbors:
            neighbors_count = max_neighbors
        neighbors[i] = neighbors_count + 2 # neighbors + hydrogens
        shift += 9

    if bonds_count:
        bonds_count /= 2

        order_count = bonds_count * 3
        if order_count % 8:
            order_count = order_count / 8 + 1
        else:
            order_count /= 8

        n = add_cls
        for i in range(0, 2 * bonds_count, 2):
            a, b, c = data[shift], data[shift + 1], data[shift + 2]
            m = mapping[a << 4 | b >> 4]
            while not connections[n]:
                n += 1
            connections[n] -= 1
            distance[n, m] = distance[m, n] = 1

            m = mapping[(b & 0x0f) << 8 | c]
            while not connections[n]:
                n += 1
            connections[n] -= 1
            distance[n, m] = distance[m, n] = 1
            shift += 3

        # floyd-warshall algo
        for k in range(add_cls, atoms_count):
            for i in range(add_cls, atoms_count):
                if i == k or distance[i, k] == 9999:
                    continue
                for j in range(add_cls, atoms_count):
                    d = distance[i, k] + distance[k, j]
                    if d < distance[i, j]:
                        distance[i, j] = d

    # reset distances to proper values
    for i in range(add_cls, atoms_count):
        for j in range(i, atoms_count):
            d = distance[i, j]
            if d == 9999:
                # set attention between subgraphs
                distance[i, j] = distance[j, i] = components_attention
            elif d > max_distance:
                distance[i, j] = distance[j, i] = max_distance + 2
            else:
                distance[i, j] = distance[j, i] = d + 2

    size = shift + order_count + 4 * cis_trans_count
    PyMem_Free(connections)
    return atoms, neighbors, distance, size
