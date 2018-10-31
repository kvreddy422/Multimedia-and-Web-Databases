Basic Operations
================

Most operations we offer return results with :class:`tf.Tensor` form, except some build-in class methods in our module.

To begin with, load in operation module:

.. code-block:: python

   import factorizer.base.ops as ops


Basic Operations with Matrices
------------------------------


Hadamard Products
^^^^^^^^^^^^^^^^^
The *Hadamard product* is the elementwise matrix product. Given matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`, both
of size :math:`\mathit{I} \times \mathit{J}`, their Hadamard product is denoted by :math:`\mathbf{A} \ast \mathbf{B}`.
The result is defined by

.. math::
   \mathbf{A} \ast \mathbf{B} =
   \left[
   \begin{matrix}
   a_{11}b_{11}                       & a_{12}b_{12}                      & \cdots  & a_{1 \mathit{J}}b_{1 \mathit{J}}\\
   a_{21}b_{21}                       & a_{22}b_{22}                      & \cdots  & a_{2 \mathit{J}}b_{2 \mathit{J}}\\
   \vdots                             & \vdots                            & \ddots  & \vdots\\
   a_{\mathit{I} 1}b_{\mathit{I} 1}   & a_{\mathit{I} 2}b_{\mathit{I} 2}  & \cdots  & a_{\mathit{I} \mathit{J}}b_{\mathit{I} \mathit{J}}
   \end{matrix}
   \right]

For instance, using matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1  & 4  & 7\\
   2  & 5  & 8\\
   3  & 6  & 9
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                         \begin{matrix}
   2 & 3 & 4\\
   2 & 3 & 4\\
   2 & 3 & 4
                         \end{matrix}
                         \right]

Using :class:`DTensor` to store matrices, :math:`\, \mathbf{A} \ast \mathbf{B}` can be performed as:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,4,7], [2,5,8],[3,6,9]]))
   >>> B = DTensor(tf.constant([[2,3,4], [2,3,4],[2,3,4]]))
   >>> result = A*B    # result is a DTensor with shape (3,3)
   >>> tf.Session().run(result.T)
   array([[ 2, 12, 28],
          [ 4, 15, 32],
          [ 6, 18, 36]], dtype=int32)

Using :class:`tf.Tensor` to store matrices, :math:`\, \mathbf{A} \ast \mathbf{B}` can be performed as:

.. code-block:: python

   >>> A = tf.constant([[1,4,7], [2,5,8],[3,6,9]])
   >>> B = tf.constant([[2,3,4], [2,3,4],[2,3,4]])
   >>> tf.Session().run(ops.hadamard([A,B]))
   array([[ 2, 12, 28],
          [ 4, 15, 32],
          [ 6, 18, 36]], dtype=int32)

:func:`hadamard` also supports the Hadamard products of more than two matrices:

.. code-block:: python

   >>> C = tf.constant(np.random.rand(3,3))
   >>> D = tf.constant(np.random.rand(3,3))
   >>> tf.Session().run(ops.hadamard([A, B, C, D], skip_matrices_index=[1]))
       # the result is equal to tf.Session().run(ops.hadamard([A, C, D]))

Kronecker Products
^^^^^^^^^^^^^^^^^^
The *Kronecker product* of matrices :math:`\, \mathbf{A} \in \mathbb{R}^{\mathit{I} \times \mathit{J}}`
and :math:`\mathbf{B} \in \mathbb{R}^{\mathit{K} \times \mathit{L}}` is denoted by :math:`\mathbf{A} \otimes \mathbf{B}`.
The result is a matrix of size :math:`(\mathit{IK}) \times (\mathit{JL})` (See Kolda's [1]_ for more details).

For example, matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1   & 2   & 3   & 4\\
   5   & 6   & 7   & 8\\
   9   & 10  & 11  & 12
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                                \begin{matrix}
   1 & 1 & 1 & 1 & 1\\
   2 & 2 & 2 & 2 & 2
                                \end{matrix}
                                \right]
To perform :math:`\mathbf{A} \otimes \mathbf{B}` with :class:`tf.Tensor` objects:

.. code-block:: python

   >>> A = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])    # the shape of A is (3, 4)
   >>> B = tf.constant([[1,1,1,1,1],[2,2,2,2,2]])    # the shape of B is (2, 5)
   >>> tf.Session().run(ops.kron([A, B]))
   # the shape of result is (6, 20)
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)

To perform :math:`\mathbf{B} \otimes \mathbf{A}`:

.. code-block:: python

   >>> tf.Session().run(ops.kron([A, B], reverse=True))
   # the shape of result is (6, 20)
   array([[ 1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8],
          [ 9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12],
          [ 2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8,  2,  4,  6,  8],
          [10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16, 10, 12, 14, 16],
          [18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24, 18, 20, 22, 24]], dtype=int32)

It might seem useless when using ``reverse=True`` to calculate the Kronecker product of two matrices, considering ``ops.kron([B, A])``
also do the same work, but it is considerable efficient to perform :math:`X_1 \otimes X_2 \otimes \cdots \otimes X_N` using ``reverse=True`` when
given a list of :class:`tf.Tensor` objects ``matrices = [X_1, X_2, ..., X_N]``:

.. code-block:: python

   >>> tf.Session().run(ops.kron(matrices, reverse=True))

If the matrices are given in :class:`DTensor` form:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))

Then :math:`\mathbf{A} \otimes \mathbf{B}` can be performed as:

.. code-block:: python

   >>> dtensor_B = DTensor(tf.constant([[1,1,1,1,1],[2,2,2,2,2]]))
   >>> tf.Session().run(A.kron(dtensor_B).T)    # A.kron(dtensor_B) returns a DTensor
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)

or

.. code-block:: python

   >>> tf_B = tf.constant([[1,1,1,1,1],[2,2,2,2,2]])
   >>> tf.Session().run(A.kron(tf_B).T)    # A.kron(tf_B) returns a DTensor
   array([[ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4],
          [ 2,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8],
          [ 5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8],
          [10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16],
          [ 9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12],
          [18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 24, 24, 24, 24, 24]], dtype=int32)



Khatri-Rao Products
^^^^^^^^^^^^^^^^^^^
The *Khatri-Rao product* can be expressed in Kronecker product form. Given matrices :math:`\mathbf{A} \in \mathbb{R}^{\mathit{I} \times \mathit{K}}`
and :math:`\mathbf{B} \in \mathbb{R}^{\mathit{J} \times \mathit{K}}` , their Khatri-Rao product is denoted by :math:`\mathbf{A} \odot \mathbf{B}`.
The result is a matrix of size :math:`(\mathit{IJ}) \times (\mathit{K})` and defined by

.. math::
   \mathbf{A} \odot \mathbf{B} =
   \left[
   \begin{matrix}
   \mathbf{a}_1 \otimes \mathbf{b}_1 &  \mathbf{a}_2 \otimes \mathbf{b}_2  & \cdots  & \mathbf{a}_\mathit{K} \otimes \mathbf{b}_\mathit{K}
   \end{matrix}
   \right]

Let's take a look at matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` defined as

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1   & 2   & 3   & 4\\
   5   & 6   & 7   & 8\\
   9   & 10  & 11  & 12
   \end{matrix}
   \right] , \quad \mathbf{B} = \left[
                                \begin{matrix}
   1 & 1 & 1 & 1\\
   2 & 2 & 2 & 2
                                \end{matrix}
                                \right]

To perform :math:`\mathbf{A} \odot \mathbf{B}`:

.. code-block:: python

   >>> A = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])    # the shape of A is (3, 4)
   >>> B = tf.constant([[1,1,1,1],[2,2,2,2]])    # the shape of B is (2, 4)
   >>> tf.Session().run(ops.khatri([A, B]))
   # the shape of the result is (6, 4)
   array([[ 1,  2,  3,  4],
          [ 2,  4,  6,  8],
          [ 5,  6,  7,  8],
          [10, 12, 14, 16],
          [ 9, 10, 11, 12],
          [18, 20, 22, 24]], dtype=int32)

:func:`khatri` function also offers ``skip_matrices_index`` to ignore specific matrices in the computation. For example, given ``matrices = [A, B, C, D]`` to
calculate :math:`\mathbf{A} \odot \mathbf{B} \odot \mathbf{D}`:

.. code-block:: python

   >>> C = tf.constant(np.random.rand(4,4))
   >>> D = tf.constant(np.random.rand(5,4))
   >>> matrices = [A, B, C, D]
   >>> tf.Session().run(ops.khatri(matrices, skip_matrices_index=[2]))
   # the shape of the result is (30, 4)

To obtain the result of :math:`\mathbf{D} \odot \mathbf{C} \odot \mathbf{B} \odot \mathbf{A}`:

.. code-block:: python

   >>> tf.Session().run(ops.khatri(matrices, reverse=True))
   # the shape of the result is (120, 4)

:class:`DTensor` class also offers class method :func:`DTensor.khatri` which accepts only one single :class:`DTensor` object or :class:`tf.Tensor` object:

.. code-block:: python

   >>> A = DTensor(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
   >>> B = tf.constant([[1,1,1,1],[2,2,2,2]])
   >>> tf.Session().run(A.khatri(B).T)
   # the shape of the result is (6, 4)
   array([[ 1,  2,  3,  4],
          [ 2,  4,  6,  8],
          [ 5,  6,  7,  8],
          [10, 12, 14, 16],
          [ 9, 10, 11, 12],
          [18, 20, 22, 24]], dtype=int32)




Basic Operations with Tensors
-----------------------------

Addition & Subtraction
^^^^^^^^^^^^^^^^^^^^^^
Given a :class:`DTensor` object, it is easy to perform addition and subtraction.

.. code-block:: python

   >>> X = DTensor(tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]))    # the shape of tensor X is (2, 2, 3)
   >>> Y = DTensor(tf.constant([[[-1,-2,-3],[-4,-5,-6]],[[-7,-8,-9],[-10,-11,-12]]]))    # the shape of tensor Y is (2, 2, 3)
   >>> sum_X_Y = X + Y    # sum_X_Y is a DTensor
   >>> sub_X_Y = X - Y    # sub_X_Y is a DTensor
   >>> tf.Session().run(sum_X_Y.T)
   array([[[0, 0, 0],
           [0, 0, 0]],

          [[0, 0, 0],
           [0, 0, 0]]], dtype=int32)
   >>> tf.Session().run(sub_X_Y.T)
   array([[[ 2,  4,  6],
           [ 8, 10, 12]],

          [[14, 16, 18],
           [20, 22, 24]]], dtype=int32)


The second operand can also be a :class:`tf.Tensor` object:

.. code-block:: python

   >>> Z = tf.constant([[[-1,-2,-3],[-4,-5,-6]],[[-7,-8,-9],[-10,-11,-12]]])    # the shape of tensor Z is (2, 2, 3)
   >>> sum_X_Z = X + Z    # sum_X_Z is a DTensor
   >>> sub_X_Z = X - Z    # sub_X_Z is a DTensor
   >>> tf.Session().run(sum_X_Z.T)
   array([[[0, 0, 0],
           [0, 0, 0]],

          [[0, 0, 0],
           [0, 0, 0]]], dtype=int32)
   >>> tf.Session().run(sub_X_Z.T)
   array([[[ 2,  4,  6],
           [ 8, 10, 12]],

          [[14, 16, 18],
           [20, 22, 24]]], dtype=int32)




Inner Products
^^^^^^^^^^^^^^
The *inner product* of two same-sized tensor :math:`\mathcal{X}, \mathcal{Y} \in \mathbb{R}^{\mathit{I}_1 \times \mathit{I}_2 \times \cdots \times \mathit{I}_N}`
is the sum of products of their entries, which can be denoted as :math:`\langle \mathcal{X} , \mathcal{Y} \rangle`.

Given tensor :math:`\mathcal{X}, \mathcal{Y} \in \mathbb{R}^\mathit{3 \times 3 \times 2}` defined by their
frontal slices:

.. math::
   X_1 =
   \left[
   \begin{matrix}
   1  & 4  & 7\\
   2  & 5  & 8\\
   3  & 6  & 9
   \end{matrix}
   \right] , \quad X_2 = \left[
                         \begin{matrix}
   10 & 13 & 16\\
   11 & 14 & 17\\
   12 & 15 & 18
                         \end{matrix}
                         \right]

.. math::
   Y_1 =
   \left[
   \begin{matrix}
   1  & 1  & 1\\
   1  & 1  & 1\\
   1  & 1  & 1
   \end{matrix}
   \right] , \quad Y_2 = \left[
                         \begin{matrix}
   1 & 1 & 1\\
   1 & 1 & 1\\
   1 & 1 & 1
                         \end{matrix}
                         \right]

.. code-block:: python

   >>> X = tf.constant(np.array([[[1,10],[4,13],[7,16]], [[2,11],[5,14],[8,17]], [[3,12],[6,15],[9,18]]]))    # the shape of X is (3, 3, 2)
   >>> Y = tf.constant(np.array([[[1,1],[1,1],[1,1]], [[1,1],[1,1],[1,1]], [[1,1],[1,1],[1,1]]]))    # the shape of Y is (3, 3, 2)

To calculate :math:`\langle \mathcal{X} , \mathcal{Y} \rangle`:

.. code-block:: python

   >>> tf.Session().run(ops.inner(X, Y))
   171



.. warning::
   Notice that :func:`ops.inner` function does not support implicit type-casting, so be careful when using tensors
   of different ``dtype`` !


Vectorization & Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The *vectorization* of a tensor is ordering the tensor into a vector. And the process transforming the vector back to the
tensor is called *reconstruction* or reshaping.

Take the tensor :math:`\mathcal{X} \in \mathbb{R}^{\mathit{3} \times \mathit{3} \times \mathit{2}}` defined before as example.

.. code-block:: python

   >>> X = tf.constant(np.array([[[1,10],[4,13],[7,16]], [[2,11],[5,14],[8,17]], [[3,12],[6,15],[9,18]]]))    # the shape of X is (3, 3, 2)
   >>> vec = ops.vectorize(X)
   >>> tf.Session().run(vec)
   array([ 1, 10,  4, 13,  7, 16,  2, 11,  5, 14,  8, 17,  3, 12,  6, 15,  9, 18])

To reconstruct the vector:

.. code-block:: python

   >>> tf.Session().run(ops.vec_to_tensor(vec,(3,3,2)))
   array([[[ 1, 10],
           [ 4, 13],
           [ 7, 16]],

          [[ 2, 11],
           [ 5, 14],
           [ 8, 17]],

          [[ 3, 12],
           [ 6, 15],
           [ 9, 18]]])


Unfolding & Folding
^^^^^^^^^^^^^^^^^^^
*Unfolding*, also known as *matricization*, is the process of reordering the elements of an *N* -way array into a matrix.
Here we call operation **mode-n matricization** as **unfolding** in default.

Let the frontal slices of :math:`\mathcal{X} \in \mathbb{R}^{\mathit{3} \times \mathit{4} \times \mathit{2}}` be

.. math::
   X_1 =
   \left[
   \begin{matrix}
   1  & 4  & 7  & 10\\
   2  & 5  & 8  & 11\\
   3  & 6  & 9  & 12
   \end{matrix}
   \right] , \quad X_2 = \left[
                         \begin{matrix}
   13 & 16 & 19 & 22\\
   14 & 17 & 20 & 23\\
   15 & 18 & 21 & 24
                         \end{matrix}
                         \right]

.. code-block:: python

   >>> X = tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])    # the shape of X is (3, 4, 2)

To get the mode-1 matricization of tensor :math:`\mathcal{X}`:

.. code-block:: python

   >>> tf.Session().run(ops.unfold(X, 0))
   array([[ 1,  4,  7, 10, 13, 16, 19, 22],
          [ 2,  5,  8, 11, 14, 17, 20, 23],
          [ 3,  6,  9, 12, 15, 18, 21, 24]], dtype=int32)

To get the mode-2 matricization of tensor :math:`\mathcal{X}`:

.. code-block:: python

   >>> tf.Session().run(ops.unfold(X, 1))
   array([[ 1,  2,  3, 13, 14, 15],
          [ 4,  5,  6, 16, 17, 18],
          [ 7,  8,  9, 19, 20, 21],
          [10, 11, 12, 22, 23, 24]], dtype=int32)

To get the mode-3 matricization of tensor :math:`\mathcal{X}`:

.. code-block:: python

   >>> tf.Session().run(ops.unfold(X, 2))
   array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
          [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], dtype=int32)

For a :class:`DTensor` object, class method :func:`DTensor.unfold` is available:

.. code-block:: python

   >>> X = DTensor(tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]]))
   >>> tf.Session().run(X.unfold(mode=0).T)    # mode-1 matricization, X.unfold(mode=0) return a DTensor
   array([[ 1,  4,  7, 10, 13, 16, 19, 22],
          [ 2,  5,  8, 11, 14, 17, 20, 23],
          [ 3,  6,  9, 12, 15, 18, 21, 24]], dtype=int32)
   






General Matricization
^^^^^^^^^^^^^^^^^^^^^
According to Kolda's [2]_, *general matricization* can flatten a high-order tensor into a matrix with size defined
with row indices and column indices.

Given tensor :math:`\mathcal{X} \in \mathbb{R}^{\mathit{I}_1 \times \mathit{I}_2 \times \cdots \times \mathit{I}_N}`,
if we want to rearrange it into a matrix with size :math:`\mathit{J}_1 \times \mathit{J}_2`,

.. math::
   where \quad \mathit{J}_1 = \prod_{k=1}^{K} \mathit{I}_{r_k} \quad and \quad \mathit{J}_2 = \prod_{\ell=1}^{L}\mathit{I}_{c_\ell}.

The set :math:`\{ r_1, \cdots, r_K \}` defines those indices that will mapped to the row indices of the resulting matrix and
the set :math:`\{ c_1, \cdots, c_L \}` defines those indices that will mapped to the column indices.

.. note::
   The order of :math:`\{ r_1, \cdots, r_K \}` or :math:`\{ c_1, \cdots, c_L \}` is not necessarily ascending or descending.

Take a look at tensor :math:`\mathcal{X}` defined as:

.. code-block:: python

   >>> X = tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])    # the shape of X is (3, 4, 2)

To mapped :math:`\mathcal{X}` into matrix of size :math:`(4 \times 3) \times 2 = 12 \times 2`:

.. code-block:: python

   >>> r_axis = [1,0]    # indices of row
   >>> c_axis = 2    # indices of column
   >>> mat = ops.t2mat(X, r_axis, c_axis)    # mat is a tf.Tensor
   >>> tf.Session().run(mat)
   array([[ 1, 13],
          [ 2, 14],
          [ 3, 15],
          [ 4, 16],
          [ 5, 17],
          [ 6, 18],
          [ 7, 19],
          [ 8, 20],
          [ 9, 21],
          [10, 22],
          [11, 23],
          [12, 24]], dtype=int32)

function :func:`ops.t2mat` can also perform *mode-n matricization* mapping indices appropriately:

To perform Kolda-type mode-2 unfolding:

.. code-block:: python

   >>> mat1 = ops.t2mat(X, 1, [2,0])

To perform LMV-type mode-2 unfolding:

.. code-block:: python

   >>> mat2 = ops.t2mat(X, 1, [0,2])

:class:`DTensor` also offers class method:

.. code-block:: python

   >>> X = DTensor(tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]]))    # the shape of X is (3, 4, 2)
   >>> mat3 = X.t2mat([1,0], 2)    # mat3 is a DTensor
   >>> tf.Session().run(mat3.T)
   array([[ 1, 13],
          [ 2, 14],
          [ 3, 15],
          [ 4, 16],
          [ 5, 17],
          [ 6, 18],
          [ 7, 19],
          [ 8, 20],
          [ 9, 21],
          [10, 22],
          [11, 23],
          [12, 24]], dtype=int32)



The *n* -mode Products
^^^^^^^^^^^^^^^^^^^^^^
The *n-mode product* of a tensor :math:`\mathcal{X} \in \mathbb{R}^{\mathit{I}_1 \times \mathit{I}_2 \times \cdots \times \mathit{I}_N}` with
 a matrix :math:`\mathbf{A} \in \mathbb{R}^{\mathit{J} \times \mathit{I}_n}` is denoted by :math:`\mathcal{X} \times_n \mathbf{A}` and
is of size :math:`\mathit{I}_1 \times \cdots \times \mathit{I}_{n-1} \times \mathit{J} \times \mathit{I}_{n+1} \times \cdots \times \mathit{I}_N`.

Let the frontal slices of :math:`\mathcal{X} \in \mathbb{R}^{\mathit{3} \times \mathit{4} \times \mathit{2}}` be

.. math::
   X_1 =
   \left[
   \begin{matrix}
   1  & 4  & 7  & 10\\
   2  & 5  & 8  & 11\\
   3  & 6  & 9  & 12
   \end{matrix}
   \right] , \quad X_2 = \left[
                         \begin{matrix}
   13 & 16 & 19 & 22\\
   14 & 17 & 20 & 23\\
   15 & 18 & 21 & 24
                         \end{matrix}
                         \right]

.. code-block:: python

   >>> X = tf.constant([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])    # the shape of X is (3, 4, 2)

And Let :math:`\mathbf{A}` be

.. math::
   \mathbf{A} =
   \left[
   \begin{matrix}
   1 & 3 & 5\\
   2 & 4 & 6
   \end{matrix}
   \right].

.. code-block:: python

   >>> A = tf.constant([[1,3,5], [2,4,6]])

Then the product :math:`\mathcal{Y} = \mathcal{X} \times_1 \mathbf{A} \in \mathbb{R}^{2 \times 4 \times 2}` is

.. math::
   Y_1 =
   \left[
   \begin{matrix}
   22  & 49  & 76  & 103\\
   28  & 64  & 100 & 136
   \end{matrix}
   \right] , \quad Y_2 = \left[
                         \begin{matrix}
   130 & 157 & 184 & 211\\
   172 & 208 & 244 & 280
                         \end{matrix}
                         \right]

Now run code below to perform the calculation:

.. code-block:: python

   >>> Y = tf.Session().run(ops.ttm(X,[A],[0]))
   >>> Y[:,:,0]    # the first frontal slice of Y
   array([[ 22,  49,  76, 103],
          [ 28,  64, 100, 136]], dtype=int32)
   >>> Y[:,:,1]    # the second frontal slice of Y
   array([[130, 157, 184, 211],
          [172, 208, 244, 280]], dtype=int32)

It is often desirable to calculate the prduct of a tensor and a sequence of matrices.
Let :math:`\mathcal{X}` be an :math:`\mathbb{R}^{\mathit{I}_1 \times \mathit{I}_2 \times \cdots \times \mathit{I}_N}` tensor,
and let :math:`\mathbf{A}^{(n)} \in \mathbb{R}^{\mathit{J}_n \times \mathit{I}_n}` for :math:`n = 1, 2, \cdots, N`. The
the sequence of products

.. math::
   \mathcal{Y} = \mathcal{X} \times_1 \mathbf{A}^{(1)} \times_2 \mathbf{A}^{(2)} \cdots \times_N \mathbf{A}^{(N)}

To perform this calculation:

.. code-block:: python

   >>> A1 = tf.constant(np.random.rand(J1,I1))
   >>> A2 = tf.constant(np.random.rand(J2,I2))
   ...
   >>> AN = tf.constant(np.random.rand(JN,IN))
   >>> X = tf.constant(np.random.rand(I1, I2, ..., IN))
   >>> seq_A = [A1, A2, ..., AN]    # map all matrices into a list
   >>> B = ops.ttm(X, seq_A, axis=range(N))
   >>> tf.Session().run(B)

If needed, arguments ``transpose`` and ``skip_matrices_index`` are also available.


Tensor Contraction
^^^^^^^^^^^^^^^^^^
The *tensor contraction* multiplies two tensors along the given axis. Let tensor :math:`\mathcal{X} \in \mathbb{R}^{\mathit{I}_1 \times \cdots \times \mathit{I}_M \times \mathit{J}_1 \times \cdots \times \mathit{J}_N}`,
and tensor :math:`\mathcal{Y} \in \mathbb{R}^{\mathit{I}_1 \times \cdots \times \mathit{I}_M  \times \mathit{K}_1 \times \cdots \times \mathit{K}_P}`,
then multiplying both tensors along the first :math:`M` modes can be denoted by :math:`\mathcal{Z} = \langle \mathcal{X} , \mathcal{Y} \rangle_{ \{ 1, \dots , M; 1, \dots, M \} }`.
And the size of :math:`\mathcal{Z}` is :math:`\mathit{J}_1 \times \cdots \times \mathit{J}_N \times \mathit{K}_1 \times \cdots \times \mathit{K}_P`. See Cichocki’s [3]_ for more details.

To perform tensor contraction:

.. code-block:: python

   >>> X = tf.constant(np.random.rand(I1, ..., IM, J1, ..., JN))
   >>> Y = tf.constant(np.random.rand(I1, ..., IM, K1, ..., KP))
   >>> Z = ops.mul(X, Y,  a_axis=[0,1,...,M-1], b_axis=[0,1,...,M-1])    # either a_axis or b_axis can also be tuple or a single integer
   >>> tf.Session().run(Z)    # Z is a tf.Tensor object

The arguments ``a_axis`` and ``b_axis`` specifying the modes of :math:`\mathcal{X}` and :math:`\mathcal{Y}` for contraction
are not consecutive necessarily, but the sizes of corresponding dimensions must be equal.

Classic matrix multiplication can also be performed with :func:`mul`:

.. code-block:: python

   >>> A = tf.constant(np.random.rand(4,5))    # matrix A with shape (4, 5)
   >>> B = tf.constant(np.random.rand(5,4))    # matrix B with shape (5, 4)
   >>> C = ops.mul(A, B, 1, 0)    # same as tf.matmul(A, B, transpose_a=False, transpose_b=False)
   >>> D = ops.mul(A, B, 0, 1)    # same as tf.matmul(A, B, transpose_a=True, transpose_b=True)

Class :class:`DTensor` also provides class method :func:`DTensor.mul`:

.. code-block:: python

   >>> X_dtensor = DTensor(np.random.rand(4,5))
   >>> Y_dtensor = DTensor(np.random.rand(5,4))
   >>> Z_dtensor = X_dtensor.mul(Y_dtensor, a_axis=1, b_axis=0)
   # same as DTensor( tf.matmul(X_dtensor.T, Y_dtensor.T, transpose_a=False, transpose_b=False) )

The argument ``tensor`` of :func:`DTensor.mul` only accepts :class:`DTensor` object.

References
----------
.. [1] Tamara G. Kolda and Brett W. Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
.. [2] Tamara G. Kolda and Brett W. Bader, "Algorithm 862: MATLAB tensor classes for fast algorithm prototyping",
       ACM Trans. Math. Softw, 32 (4): 635-653 (2006)
.. [3] Cichocki, Andrzej. "Era of big data processing: A new approach via tensor networks and tensor decompositions."
     arXiv preprint arXiv:1403.2048 (2014).




