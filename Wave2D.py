import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        xij, yij = np.meshgrid(x, y, sparse=sparse, indexing="ij")
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.sqrt((np.pi * self.mx)**2 + (np.pi * self.my)**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        U0 = np.zeros((N,N))
        U1 = np.zeros((N,N))
        xij, yij = self.create_mesh(N)
        U0 = self.ue(mx, my)
        U0 = sp.lambdify((x, y, t), U0)(xij,yij, 0)
        U1 = self.ue(mx, my)
        U1 = sp.lambdify((x, y, t), U1)(xij,yij, self.dt)
        U0 = self.apply_bcs(U0)
        U1 = self.apply_bcs(U1)
        return U0, U1

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = self.ue(self.mx, self.my)
        ue = sp.lambdify((x, y, t), ue)(*self.create_mesh(u.shape[0]-1), t0)
        return np.sqrt(np.sum((u - ue)**2) * self.dx**2)

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        N = u.shape[0]
        u[-1,:] = u [0,:] = u[:,-1] = u[:,0] = 0
        return u

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.cfl = cfl
        self.mx = mx
        self.my = my
        self.dx = 1 / N
        xij, yij = self.create_mesh(N)
        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        Unm1, Un = self.initialize(N, mx, my)
        dict_ts = {}

        for t in range(2, Nt):
            Unp1[:] = 2 * Un - Unm1 + (c * self.dt)**2 * (self.D2(N) @ Un + Un @ self.D2(N).T) 
            Unp1 = self.apply_bcs(Unp1)
            if store_data>0:
                dict_ts[t] = Unp1.copy()
            Unm1 = Un
            Un = Unp1
            
        if store_data>0:
            return dict_ts

        elif(store_data == -1):
            l2_error = self.l2_error(Un, 0)
            return (self.dx, l2_error)

        else:
            raise ValueError("The given value for store_data is wrong")

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix
        
            Parameters
            ----------
            N : int
                The number of uniform intervals in each direction
            Returns
            -------
            D : scipy sparse LIL matrix
                The second order differentiation matrix
        """
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :2] = -2, 2
        D2[-1, -2:] = 2, -2
        return D2

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        return u


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    print(r[-1]-2)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(m=5,mx=3, my=3)
    print(r)
    assert abs(r[-1] - 2) < 0.05, r


def test_exact_wave2d():
    raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    print("All tests passed!")