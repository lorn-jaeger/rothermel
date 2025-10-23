from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import cProfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--F", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prows", type=int, default=0)
    ap.add_argument("--pcols", type=int, default=0)
    ap.add_argument("--out_prefix", type=str, default="mpi")
    return ap.parse_args()

def factor_size(n):
    import math
    pcols = int(math.sqrt(n))
    while pcols > 1 and n % pcols != 0:
        pcols -= 1
    prows = n // pcols
    return prows, pcols

def setup_cart(prows, pcols):
    ix = rank % pcols
    iy = rank // pcols
    left  = rank-1 if ix > 0 else MPI.PROC_NULL
    right = rank+1 if ix < pcols-1 else MPI.PROC_NULL
    up    = rank-pcols if iy > 0 else MPI.PROC_NULL
    down  = rank+pcols if iy < prows-1 else MPI.PROC_NULL
    return ix, iy, left, right, up, down

def local_shape(nx, ny, prows, pcols):
    assert nx % pcols == 0 and ny % prows == 0, "wrong dims"
    nx_local = nx // pcols  
    ny_local = ny // prows  
    return nx_local, ny_local

def init_phi(nx, ny, dx, dy, ix, iy, nx_local, ny_local):
    x0 = ix * nx_local * dx
    y0 = iy * ny_local * dy
    x = x0 + dx*np.arange(-1, nx_local+1)   
    y = y0 + dy*np.arange(-1, ny_local+1)
    X, Y = np.meshgrid(x, y, indexing='ij') 
    xc, yc = (nx*dx)/2.0, (ny*dy)/2.0
    phi = np.sqrt((X - xc)**2 + (Y - yc)**2) - 10.0
    return phi

def halo_exchange(phi, left, right, up, down):
    send_row_to_left  = phi[1, 1:-1].copy()
    recv_row_from_right = np.empty_like(send_row_to_left)
    comm.Sendrecv(send_row_to_left, dest=left,  sendtag=101,
                  recvbuf=recv_row_from_right, source=right, recvtag=101)
    if right != MPI.PROC_NULL:
        phi[-1, 1:-1] = recv_row_from_right
    else:
        phi[-1, 1:-1] = phi[-2, 1:-1]

    send_row_to_right = phi[-2, 1:-1].copy()
    recv_row_from_left = np.empty_like(send_row_to_right)
    comm.Sendrecv(send_row_to_right, dest=right, sendtag=102,
                  recvbuf=recv_row_from_left, source=left,  recvtag=102)
    if left != MPI.PROC_NULL:
        phi[0, 1:-1] = recv_row_from_left
    else:
        phi[0, 1:-1] = phi[1, 1:-1]

    send_col_to_up = phi[1:-1, 1].copy()
    recv_col_from_down = np.empty_like(send_col_to_up)
    comm.Sendrecv(send_col_to_up, dest=up,   sendtag=201,
                  recvbuf=recv_col_from_down, source=down, recvtag=201)
    if down != MPI.PROC_NULL:
        phi[1:-1, -1] = recv_col_from_down
    else:
        phi[1:-1, -1] = phi[1:-1, -2]

    send_col_to_down = phi[1:-1, -2].copy()
    recv_col_from_up = np.empty_like(send_col_to_down)
    comm.Sendrecv(send_col_to_down, dest=down, sendtag=202,
                  recvbuf=recv_col_from_up,   source=up,   recvtag=202)
    if up != MPI.PROC_NULL:
        phi[1:-1, 0] = recv_col_from_up
    else:
        phi[1:-1, 0] = phi[1:-1, 1]

def step_upwind(phi, F, dx, dy, dt):
    new_phi = phi.copy()

    c  = phi[1:-1, 1:-1]  
    xm = phi[0:-2, 1:-1] 
    xp = phi[2:  , 1:-1] 
    ym = phi[1:-1, 0:-2]
    yp = phi[1:-1, 2:  ]

    dxb = (c - xm) / dx   
    dxf = (xp - c) / dx   
    dyb = (c - ym) / dy   
    dyf = (yp - c) / dy   

    grad_x_sq = np.maximum(dxb, 0.0)**2 + np.minimum(dxf, 0.0)**2
    grad_y_sq = np.maximum(dyb, 0.0)**2 + np.minimum(dyf, 0.0)**2
    grad_phi  = np.sqrt(grad_x_sq + grad_y_sq)

    new_phi[1:-1, 1:-1] = c - dt * F * grad_phi
    return new_phi

def main():
    args = parse_args()
    if args.prows == 0 or args.pcols == 0:
        prows, pcols = factor_size(size)
    else:
        prows, pcols = args.prows, args.pcols
        assert prows*pcols == size, "remember prows*pcols"

    nx_local, ny_local = local_shape(args.nx, args.ny, prows, pcols)
    ix, iy, left, right, up, down = setup_cart(prows, pcols)

    phi = init_phi(args.nx, args.ny, args.dx, args.dy, ix, iy, nx_local, ny_local)

    for _ in range(args.steps):
        halo_exchange(phi, left, right, up, down)
        phi = step_upwind(phi, args.F, args.dx, args.dy, args.dt)

    core = phi[1:-1, 1:-1].copy()  
    sendbuf = core.ravel()

    if rank == 0:
        full = np.empty((args.nx, args.ny), dtype=np.float64)
        recvbuf = np.empty_like(sendbuf)
        full[ix*nx_local:(ix+1)*nx_local, iy*ny_local:(iy+1)*ny_local] = core
        for r in range(1, size):
            comm.Recv(recvbuf, source=r, tag=99)
            rx = r % pcols
            ry = r // pcols
            full[rx*nx_local:(rx+1)*nx_local, ry*ny_local:(ry+1)*ny_local] = recvbuf.reshape(nx_local, ny_local)
    else:
        comm.Send(sendbuf, dest=0, tag=99)

    if rank == 0:
        np.save(f"{args.out_prefix}_phi_final.npy", full)
        plt.figure()
        plt.imshow(full.T, origin='lower', cmap='coolwarm')
        plt.colorbar(label='phi')
        plt.contour(full.T, levels=[0.0], colors='k', linewidths=2)
        plt.title(f"Final perimeter (phi=0), steps={args.steps}, F={args.F}")
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_final.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.dump_stats(f"profile_rank{rank}.prof")

