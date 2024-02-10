"""
Generates simulated HeSE data for hopping on a lattice based on AlPdMn
with random kinetic parameters.
"""


import itertools
import time
import numpy as np

import quasicrystal_surface
import particle


# Generate the lattice
print("Generating lattice...")
latt = quasicrystal_surface.make_lattice(2000)


# Define site and jump categorisation
print("Categorising jumps...")

# Define functions to classify sites and jumps
point_classifier = quasicrystal_surface.point_classifier
def jump_classifier(p,q):
	t = quasicrystal_surface.jump_classifier(p,q)
	sorted_sites = sorted(t[:2])
	return sorted_sites[0], sorted_sites[1], t[3], t[4]

# Determine the complete sets of types of each
site_types = set(
	point_classifier(p)
	for p in latt.points.values() if latt.is_in_defined_region(p.position)
)
site_type_counts = {t : 0 for t in site_types}
for p in latt.points.values():
	if latt.is_in_defined_region(p.position):
		site_type_counts[point_classifier(p)] += 1
jump_types = set(
	jump_classifier(p,q)
	for p in latt.points.values() if latt.is_in_defined_region(p.position)
	for q in p.nrst_neighbours if latt.is_in_defined_region(q.position)
)
rate_params_types = set(t[2:] for t in jump_types)

print("\nSite types and their frequencies:")
print(site_type_counts)
print("\nRate parameter categories:")
print(rate_params_types)


# Generate parameters at random
print("Generating random kinetic parameters...")
rng = np.random.default_rng()
# Determine what site and jump types we need parameters for
# Generate site probabilities uniformly between 1 and 10, then normalise
site_probabilities = {t : rng.uniform(1,10) for t in site_types}
norm_fact = (
	sum(site_type_counts[t] for t in site_types)
	/ sum(site_type_counts[t] * site_probabilities[t] for t in site_types)
)
site_probabilities = {t :  norm_fact * site_probabilities[t] for t in site_types}
# Generate rate parameters uniformly between .0001 and .001
rate_params = {t : rng.uniform(.0001,.001) for t in rate_params_types}
# Save these parameters to a text file
with open("kinetic_parameters.txt", "w") as f:
	f.write(
		"Site probabilities:\n"
		+ "".join(f"    {t:<20}{site_probabilities[t]}\n" for t in site_types)
		+ "\nRate parameters:\n"
		+ "".join(f"    {str(t):<50}{rate_params[t]}\n" for t in rate_params_types)
		+ "\n"
	)


# Perform the hopping simulation
print("\n\nSimulating hopping:")
max_rate = max(
	rate_param / site_prob
	for rate_param, site_prob in itertools.product(
		rate_params.values(), site_probabilities.values()
	)
)
def hopping_algorithm(point, rand_num):
	if rand_num > 10 * max_rate:
		return point
	cumulative_rate_total = 0
	for nbr in point.nrst_neighbours:
		if not latt.is_in_defined_region(nbr.position):
			raise RuntimeError(
				"Particle has left the region where the lattice is properly defined"
			)
		try:
			nbr_rate = (
				rate_params[jump_classifier(point, nbr)[2:]]
				/ site_probabilities[point_classifier(point)]
			)
		except KeyError:
			print(
				f"Warning: simulation encountered an unrecognised jump "
				+ f"type with key {jump_classifier(point, nbr)}"
			)
			continue
		cumulative_rate_total += nbr_rate
		if cumulative_rate_total > 1:
			raise ValueError(
				f"The simulation time step is too long for the specified "
				+ f"kinetic parameters."
			)
		elif cumulative_rate_total > rand_num:
			return nbr
	return point
part = particle.Particle(latt, latt.get_points_near([0,0], 10)[0], hopping_algorithm)
trajs = part.simulate_ensemble(1000000, 100, print_progress=True)



# Calculate simulated measurements with noise
print("\n\nGenerating incoherent scattering measurements:")
grid_size = .05
inchrnt_k_grid = np.mgrid[
	0 : 3+grid_size/2 : grid_size,
	-.1 : 3+grid_size/2 : grid_size
].reshape(2,-1).T
inchrnt_k_vects = np.array([
	k for k in inchrnt_k_grid
	if (
		(k[1] == 0 or (k[0] != 0 and np.arctan((k[1]-.1) / k[0]) <= np.pi/10))
		and np.square(k).sum() <= 9
	)
])
inchrnt_sample_ts = [2000, 3000, 4000]
inchrnt_ISF_measurements = trajs.noisy_ISF_snapshots(
	inchrnt_k_vects, inchrnt_sample_ts, 10000, True
)
print(f"(total samples: {len(inchrnt_k_vects) * len(inchrnt_sample_ts)})")
np.savetxt("incoherent_measurements.txt", inchrnt_ISF_measurements)


print("\n\nGenerating coherent scattering measurements:")
grid_size = .035
chrnt_k_grid = np.mgrid[
	0 : 3+grid_size/2 : grid_size,
	0 : 3+grid_size/2 : grid_size
].reshape(2,-1).T
chrnt_k_vects = [
	k for k in chrnt_k_grid
	if (
		(k[1] == 0 or (k[0] != 0 and np.arctan(k[1] / k[0]) <= np.pi/10))
		and np.square(k).sum() <= 9
	)
]
chrnt_sample_ts = [50, 100, 150, 225, 300, 400, 500, 700, 900, 1200]
chrnt_ISF_measurements = trajs.noisy_ISF_snapshots(
	chrnt_k_vects, chrnt_sample_ts, 10000, True
)
print(f"(total samples: {len(chrnt_k_vects) * len(chrnt_sample_ts)})")
np.savetxt("coherent_measurements.txt", chrnt_ISF_measurements)

