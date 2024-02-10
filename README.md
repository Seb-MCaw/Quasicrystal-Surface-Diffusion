Quasicrystal Surface Diffusion
==============================

The core modules written for my master's project, which implement a simplified
simulation of the diffusion of adsorbates on the surface of quasicrystals and
produce the spectroscopic data that would be expected therefrom.

The full details of the physics will likely not make sense without the project
report, but the basic procedure implemented here is fairly straightforward:

- A Penrose tiling is used to construct a lattice representation of a plausible 
quasicrystal surface
- An ensemble of simulations is performed for the movement of a species adsorbed
onto the surface, using a simple model based on discrete jumps between lattice
sites
- The pair correlation function for the diffusion statistics is constructed
and spatially Fourier transformed to produce the intermediate scattering
function, as would be measured by helium-3 spin-echo spectroscopy

A collection of further functions is provided to analyse the resulting data
using the analytic approximations which formed the main results of the project.
