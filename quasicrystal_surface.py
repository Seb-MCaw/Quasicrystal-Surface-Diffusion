"""
Implements a lattice loosely based on icosahedral AlPdMn as a lattice.Lattice
object.
"""


import numpy as np
from scipy.constants import golden as GOLDEN_RATIO

import penrose_tilings
import isf_analysis


_lng_jumps = isf_analysis.evenly_spaced_vectors(10, 7.8, np.pi/2)
_med_jumps = isf_analysis.evenly_spaced_vectors(10, 7.8 / GOLDEN_RATIO, np.pi/2)
_sht_jumps = isf_analysis.evenly_spaced_vectors(10, 7.8 / GOLDEN_RATIO**2, np.pi/2)

_id_lng_jmp = isf_analysis.identify_jump_vect(_lng_jumps, 1e-3)
_id_med_jmp = isf_analysis.identify_jump_vect(_med_jumps, 1e-3)
_id_sht_jmp = isf_analysis.identify_jump_vect(_sht_jumps, 1e-3)


def identify_jump_vect(start_lp, end_lp):
	"""
	Returns two values: first an integer in [0,9] which specifies the direction
	of the jump vect, and second one of the strings "L" (long), "M" (medium)
	or "S" (short) to specify which of the three lengths the jump has.
	"""
	dir = _id_lng_jmp(start_lp, end_lp)
	if dir is not None:
		return dir, "L"
	dir = _id_med_jmp(start_lp, end_lp)
	if dir is not None:
		return dir, "M"
	dir = _id_sht_jmp(start_lp, end_lp)
	if dir is not None:
		return dir, "S"
	raise ValueError("A jump didn't match any of those permitted")


def make_lattice(radius, long_jump_length=7.8):
	"""
	Returns a randomly generated example of the model quasicrystal lattice
	with all jumps present as a lattice.Lattice
	
	Every point also has the additional attributes lng_nbrs, med_nbrs
	and sht_nbrs defined, containing lists of the integers corresponding
	to the directions in which the exist jumps of the corresponding lengths
	leading away from the point.
	"""
	# Generate the points only lattice (removing any duplicates)
	latt = penrose_tilings.new_tiling(
		radius,
		"APM",
		long_jump_length,
		1e10*np.random.random(2),
		1.6e10
	).to_lattice()
	for p in list(latt.points.values()):
		close_points = latt.get_points_near(p.position, 1e-3)
		assert len(close_points) > 0
		if len(close_points) > 1:
			for close_point in close_points:
				if close_point is not p:
					latt.remove_point(close_point)
	# Join points with all possible jumps
	latt.link_points_separated_by(_lng_jumps + _med_jumps + _sht_jumps)
	# Remove long jumps when shorter ones are possible in the same direction
	# Take the opportunity to also define the lng_nbrs, med_nbrs amd sht_nbrs
	# attributes
	for p in latt.points.values():
		lng_nbrs, med_nbrs, sht_nbrs = [], [], []
		# Have to copy nrst_neighbours otherwise removing items confuses the loop
		for q in p.nrst_neighbours.copy():
			dir, typ = identify_jump_vect(p,q)
			if typ == "S":
				sht_nbrs.append(dir)
				if dir in med_nbrs:
					med_lp = next(
						nb for nb in p.nrst_neighbours if _id_med_jmp(p, nb) == dir
					)
					med_lp.nrst_neighbours.remove(p)
					p.nrst_neighbours.remove(med_lp)
					med_nbrs.remove(dir)
				if dir in lng_nbrs:
					lng_lp = next(
						nb for nb in p.nrst_neighbours if _id_lng_jmp(p, nb) == dir
					)
					lng_lp.nrst_neighbours.remove(p)
					p.nrst_neighbours.remove(lng_lp)
					lng_nbrs.remove(dir)
			elif typ == "M":
				if dir in sht_nbrs:
					q.nrst_neighbours.remove(p)
					p.nrst_neighbours.remove(q)
				else:
					med_nbrs.append(dir)
				if dir in lng_nbrs:
					lng_lp = next(
						nb for nb in p.nrst_neighbours if _id_lng_jmp(p, nb) == dir
					)
					lng_lp.nrst_neighbours.remove(p)
					p.nrst_neighbours.remove(lng_lp)
					lng_nbrs.remove(dir)
			elif typ == "L":
				if dir in sht_nbrs or dir in med_nbrs:
					q.nrst_neighbours.remove(p)
					p.nrst_neighbours.remove(q)
				else:
					lng_nbrs.append(dir)
			else:
				print(tuple(p.position))
				print(tuple(q.position))
				print(dir, typ)
				raise ValueError("Unrecognised jump type")
		p.lng_nbrs, p.med_nbrs, p.sht_nbrs = lng_nbrs, med_nbrs, sht_nbrs
	# Return the completed lattice
	return latt


def point_classifier(lattice_point):
	"""
	Returns a string identifying the type of a site on the lattice, based
	on the nearest neighbours in each direction.
	
	Points that are rotations or reflections of one another return the
	same string.
	"""
	lp = lattice_point
	if hasattr(lp, "site_type"):
		return lp.site_type
	else:
		s = "".join(
			(
				"S" if i in lp.sht_nbrs
				else ("M" if i in lp.med_nbrs
				else ("L" if i in lp.lng_nbrs
				else "_"
			)))
			for i in range(10)
		)
		type_string = sorted(
			[s[i:] + s[:i] for i in range(10)]
			+ [(s[i:] + s[:i])[::-1] for i in range(10)]
		)[0]
		lp.site_type = type_string
		return type_string


def jump_classifier(start, end):
	"""
	Return a tuple identifying the important characteristics of a specific
	jump between sites on the lattice.
	
	The arguments are the lattice.LatticePoint objects corresponding to
	the jump's start and end points.
	
	The returned tuple contains four characteristics (in order):
		0   The type string of the start site
		1   The type string of the end site
		2   An integer representing the direction of the jump (as returned
		    by identify_jump_vect())
		3   The length of the jump (the characters "L", "M" or "S")
		4   A string representing the presence of any other sites adjacent
		    to the jump, within a perpendicular distance to the jump's
		    direction equal to the jump's length (this string is invariant
		    under reflection and the exchange start <--> end)
	"""
	start_type = point_classifier(start)
	end_type = point_classifier(end)
	jump_dir, jump_type = identify_jump_vect(start, end)
	start_adj_jumps = (
		"S" if (jump_dir+2)%10 in start.sht_nbrs
		else ("M" if (jump_dir+1)%10 in start.med_nbrs
		else ("L" if (jump_dir+1)%10 in start.lng_nbrs
		else "_"
	)))
	start_adj_jumps += (
		"S" if (jump_dir-2)%10 in start.sht_nbrs else
		("M" if (jump_dir-1)%10 in start.med_nbrs else
		("L" if (jump_dir-1)%10 in start.lng_nbrs else "_"
	)))
	end_adj_jumps = (
		"S" if (jump_dir+3)%10 in end.sht_nbrs else
		("M" if (jump_dir+4)%10 in end.med_nbrs else
		("L" if (jump_dir+4)%10 in end.lng_nbrs else "_"
	)))
	end_adj_jumps += (
		"S" if (jump_dir-3)%10 in end.sht_nbrs else
		("M" if (jump_dir-4)%10 in end.med_nbrs else
		("L" if (jump_dir-4)%10 in end.lng_nbrs else "_"
	)))
	if jump_type == "M" or jump_type == "S":
		start_adj_jumps = start_adj_jumps.replace("L", "_")
		end_adj_jumps = end_adj_jumps.replace("L", "_")
	if jump_type == "S":
		start_adj_jumps = start_adj_jumps.replace("M", "_")
		end_adj_jumps = end_adj_jumps.replace("M", "_")
	return (
		start_type,
		end_type,
		jump_dir,
		jump_type,
		"".join(sorted((
			"".join(sorted(start_adj_jumps)),
			"".join(sorted(end_adj_jumps))
		)))
	)


