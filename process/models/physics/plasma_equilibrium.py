"""Fixed-boundary plasma equilibrium via optional veqpy integration."""

import os
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np

from process.core import constants
from process.core.exceptions import ProcessValueError
from process.core.model import Model
from process.models.physics.profiles import NeProfile, TeProfile

_veqpy: Any = None

logger = getLogger(__name__)


def _import_veqpy():
    """Import veqpy on demand; site-packages install needs a writable numba cache dir."""
    global _veqpy
    if _veqpy is None:
        numba_cache = Path(__file__).resolve().parent / ".numba_cache"
        numba_cache.mkdir(exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
        import veqpy as veq  # noqa: PLC0415

        _veqpy = veq
    return _veqpy


class PlasmaEquilibrium(Model):
    """Solve veqpy equilibria; results are stored on ``data.physics`` when bound."""

    def __init__(
        self,
        alphaj,
        i_plasma_pedestal,
        alphan,
        alphat,
        tbeta,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        radius_plasma_pedestal_density_norm,
        radius_plasma_pedestal_temp_norm,
        b_plasma_toroidal_on_axis,
        rminor,
        rmajor,
        kappa,
        triang,
    ):
        self.alphaj = alphaj
        self.i_plasma_pedestal = i_plasma_pedestal
        self.alphan = alphan
        self.alphat = alphat
        self.tbeta = tbeta
        self.nd_plasma_pedestal_electron = nd_plasma_pedestal_electron
        self.nd_plasma_separatrix_electron = nd_plasma_separatrix_electron
        self.temp_plasma_pedestal_kev = temp_plasma_pedestal_kev
        self.temp_plasma_separatrix_kev = temp_plasma_separatrix_kev
        self.radius_plasma_pedestal_density_norm = radius_plasma_pedestal_density_norm
        self.radius_plasma_pedestal_temp_norm = radius_plasma_pedestal_temp_norm
        self.rminor = rminor
        self.rmajor = rmajor
        self.kappa = kappa
        self.delta = triang
        self.b_toroidal_rmajor = b_plasma_toroidal_on_axis

    def run(self):
        """PlasmaEquilibrium is invoked on demand, not in the main model loop."""

    def output(self):
        """PlasmaEquilibrium has no dedicated output block."""

    @staticmethod
    def solve_equilibrium_veqpy(
        pres_thermal_plasma_array,
        rho_pres_array,
        j_toroidal_array,
        rho_j_array,
        current_plasma_total_ampere,
        rminor,
        rmajor,
        kappa,
        delta,
        b_toroidal_rmajor,
    ):
        """Build and solve a veqpy equilibrium for given 1D profiles and Ip.

        Raises
        ------
        RuntimeError
            If the veqpy solver does not converge.
        """
        veq = _import_veqpy()
        nrho = 51
        rho = np.linspace(0.0, 1.0, nrho)
        p = np.interp(rho, rho_pres_array, pres_thermal_plasma_array)
        jtor = np.interp(rho, rho_j_array, j_toroidal_array)

        topology = veq.KernelTopology(
            h_count=3,
            v_count=0,
            kappa_count=6,
            psin_count=0,
            F_count=0,
            c_counts=(),
            s_counts=(3,),
            Nr=16,
            Nt=16,
            route="PJ1",
            coordinate="rho",
            nodes="uniform",
            constraint="ip",
            sample_count=nrho,
        )

        kernel = veq.build(
            topology=topology,
            recipe=veq.KernelRecipe(backend="numba"),
            config=veq.KernelConfig(initial="cold"),
        )
        boundary = veq.KernelBoundary(
            a=rminor,
            R0=rmajor,
            Z0=0.0,
            B0=b_toroidal_rmajor,
            ka=kappa,
            s_offsets=(float(np.arcsin(np.clip(delta, -1.0, 1.0))),),
        )
        source = veq.KernelSource(
            p=p, jtor=jtor, Ip=float(current_plasma_total_ampere)
        )
        result = kernel.solve(boundary=boundary, source=source)
        if not result.success:
            raise RuntimeError(
                "veqpy equilibrium solve failed: "
                f"residual={result.raw_norm:.3e}, nfev={result.nfev}"
            )
        return kernel.build_equilibrium()

    @staticmethod
    def get_volume_average(eq, rho_profile, profile):
        """Volume average of a 1D profile f(rho) using equilibrium geometry."""
        profile_on_grid = np.interp(eq.rho, rho_profile, profile)
        integrand = profile_on_grid[:, None] * eq.R * eq.J
        volume = eq.grid.integrate(eq.R * eq.J)
        return float(eq.grid.integrate(integrand) / volume)

    @staticmethod
    def nd_profile(
        rho,
        i_plasma_pedestal,
        alphan,
        ne_ped,
        ne_sep,
        ne_axis,
        rho_nd_ped,
    ):
        """Electron density vs rho (parabolic or HELIOS-style pedestal)."""
        ne = np.zeros_like(rho)
        if i_plasma_pedestal == 0:
            return ne_axis * (1 - rho**2) ** alphan
        if ne_axis < ne_ped:
            logger.info(
                "NPROFILE: density pedestal is higher than core density. %s, %s",
                ne_ped,
                ne_axis,
            )
        rho_index = rho <= rho_nd_ped
        ne[rho_index] = (
            ne_ped
            + (ne_axis - ne_ped)
            * (1 - (rho[rho_index] / rho_nd_ped) ** 2) ** alphan
        )
        ne[~rho_index] = ne_sep + (ne_ped - ne_sep) * (1 - rho[~rho_index]) / (
            1 - rho_nd_ped
        )
        return ne

    @staticmethod
    def te_profile(
        rho,
        i_plasma_pedestal,
        alphat,
        tbeta,
        te_ped,
        te_sep,
        te_axis,
        rho_temp_ped,
    ):
        """Electron temperature vs rho (parabolic or HELIOS-style pedestal)."""
        if i_plasma_pedestal == 0:
            return np.maximum(te_axis * (1 - rho**2) ** alphat, 1e-8)
        if te_axis < te_ped:
            logger.info(
                "TPROFILE: temperature pedestal is higher than core temperature. %s, %s",
                te_ped,
                te_axis,
            )
        te = np.zeros_like(rho)
        rho_index = rho <= rho_temp_ped
        te[rho_index] = (
            te_ped
            + (te_axis - te_ped)
            * (1 - (rho[rho_index] / rho_temp_ped) ** tbeta)
            ** alphat
        )
        te[~rho_index] = te_sep + (te_ped - te_sep) * (1 - rho[~rho_index]) / (
            1 - rho_temp_ped
        )
        if (te < 0).any():
            raise ProcessValueError("Negative temperature in plasma profile")
        return te

    def iterate_equilibrium(self, ne_axis, te_axis, f_pres_ie, current):
        """One veqpy solve for given axis ne/te; return volume averages and q95."""
        rho = np.linspace(0.0, 1.0, 101)
        j_toroidal_array = (1.0 - rho**2) ** self.alphaj
        ne = self.nd_profile(
            rho=rho,
            i_plasma_pedestal=self.i_plasma_pedestal,
            alphan=self.alphan,
            ne_ped=self.nd_plasma_pedestal_electron,
            ne_sep=self.nd_plasma_separatrix_electron,
            ne_axis=ne_axis,
            rho_nd_ped=self.radius_plasma_pedestal_density_norm,
        )
        te = self.te_profile(
            rho=rho,
            i_plasma_pedestal=self.i_plasma_pedestal,
            alphat=self.alphat,
            tbeta=self.tbeta,
            te_ped=self.temp_plasma_pedestal_kev,
            te_sep=self.temp_plasma_separatrix_kev,
            te_axis=te_axis,
            rho_temp_ped=self.radius_plasma_pedestal_temp_norm,
        )
        pres = ne * te * constants.KILOELECTRON_VOLT * (1.0 + f_pres_ie)
        eq = self.solve_equilibrium_veqpy(
            pres_thermal_plasma_array=pres,
            rho_pres_array=rho,
            j_toroidal_array=j_toroidal_array,
            rho_j_array=rho,
            current_plasma_total_ampere=current,
            rminor=self.rminor,
            rmajor=self.rmajor,
            kappa=self.kappa,
            delta=self.delta,
            b_toroidal_rmajor=self.b_toroidal_rmajor,
        )
        ne_vol_avg = self.get_volume_average(eq, rho, ne)
        te_vol_avg = self.get_volume_average(eq, rho, te)
        q95 = float(np.interp(0.95, eq.psin, eq.q))
        return ne_vol_avg, te_vol_avg, q95, eq

    def solve_axis_for_volume_averages(
        self,
        f_pres_ie,
        ne_vol_avg_target,
        te_vol_avg_target,
        q95_target,
        current,
        ne_axis=None,
        te_axis=None,
        tol=1.0e-3,
        max_iter=25,
    ):
        """Iterate axis ne/te (and adjust Ip) until ⟨n_e⟩, ⟨T_e⟩, q95 match targets.

        Returns
        -------
        tuple
            (ne_axis, te_axis, ne_vol_avg, te_vol_avg, q95, current, eq)

        Raises
        ------
        RuntimeError
            If axis values fail to converge within ``max_iter``.
        """
        eq = None
        ne_vol_avg = te_vol_avg = q95 = 0.0

        if ne_axis is None:
            ne_axis = NeProfile.ncore(
                self.radius_plasma_pedestal_density_norm,
                self.nd_plasma_pedestal_electron,
                self.nd_plasma_separatrix_electron,
                ne_vol_avg_target,
                self.alphan,
            )
        if te_axis is None:
            te_axis = TeProfile.tcore(
                self.radius_plasma_pedestal_temp_norm,
                self.temp_plasma_pedestal_kev,
                self.temp_plasma_separatrix_kev,
                te_vol_avg_target,
                self.alphat,
                self.tbeta,
            )

        for _ in range(max_iter):
            ne_vol_avg, te_vol_avg, q95, eq = self.iterate_equilibrium(
                ne_axis, te_axis, f_pres_ie, current
            )
            ne_err = abs(ne_vol_avg - ne_vol_avg_target) / ne_vol_avg_target
            te_err = abs(te_vol_avg - te_vol_avg_target) / te_vol_avg_target
            q95_err = abs(q95 - q95_target) / q95_target
            if ne_err <= tol and te_err <= tol and q95_err <= tol:
                return ne_axis, te_axis, current, eq

            if ne_vol_avg > 0.0:
                ne_axis *= ne_vol_avg_target / ne_vol_avg
            if te_vol_avg > 0.0:
                te_axis *= te_vol_avg_target / te_vol_avg
            if q95 > 0.0:
                current *= q95 / q95_target

            if ne_axis < self.nd_plasma_pedestal_electron:
                logger.warning(
                    "Clamping ne_axis from %g to pedestal %g",
                    ne_axis,
                    self.nd_plasma_pedestal_electron,
                )
                ne_axis = self.nd_plasma_pedestal_electron
            if te_axis < self.temp_plasma_pedestal_kev:
                logger.warning(
                    "Clamping te_axis from %g to pedestal %g keV",
                    te_axis,
                    self.temp_plasma_pedestal_kev,
                )
                te_axis = self.temp_plasma_pedestal_kev

        raise RuntimeError(
            "Failed to converge ne_axis/te_axis/q95: "
            f"ne={ne_vol_avg:.4e} (target {ne_vol_avg_target:.4e}), "
            f"te={te_vol_avg:.4f} (target {te_vol_avg_target:.4f}), "
            f"q95={q95:.4f} (target {q95_target:.4f})"
        )

