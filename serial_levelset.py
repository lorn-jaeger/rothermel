import numpy as np
import argparse
import cProfile


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=3000)
    ap.add_argument("--ny", type=int, default=3000)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out_prefix", type=str, default="serial")
    return ap.parse_args()


def main():
    args = parse_args()

    nx, ny = args.nx, args.ny
    dx, dy = args.dx, args.dy
    dt = args.dt
    steps = args.steps
    F = 1.0

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    phi = np.sqrt((X - nx * dx / 2)**2 + (Y - ny * dy / 2)**2) - 10.0

    def step(phi):
        phi_x_fwd = (np.roll(phi, -1, axis=0) - phi) / dx
        phi_x_bwd = (phi - np.roll(phi, 1, axis=0)) / dx
        phi_y_fwd = (np.roll(phi, -1, axis=1) - phi) / dy
        phi_y_bwd = (phi - np.roll(phi, 1, axis=1)) / dy

        grad_x = np.where(phi_x_bwd > 0, phi_x_bwd, 0.0)**2 + np.where(phi_x_fwd < 0, phi_x_fwd, 0.0)**2
        grad_y = np.where(phi_y_bwd > 0, phi_y_bwd, 0.0)**2 + np.where(phi_y_fwd < 0, phi_y_fwd, 0.0)**2
        grad_phi = np.sqrt(grad_x + grad_y)

        return phi - dt * F * grad_phi

    for _ in range(steps):
        phi = step(phi)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.dump_stats("serial.prof")
