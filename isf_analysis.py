"""
Miscellaneous functions pertaining to the finite jump range approximation
to intermediate scattering functions.
"""


import itertools

import numpy as np
import matplotlib.pyplot as plt


def ISF_init_deriv_hop_rate_coeffs(n, jump_exists_probs, jump_vects, vect_eq_rtol=1e-9):
	"""
	Returns the predicted coefficients of the hopping rates in the expressions
	for the amplitudes of the complex exponential components of the value
	of the nth time derivative of the ISF evaluated at t=0.

	The value of the nth derivative at t=0 is a linear combination of complex
	exponential components of the form A*exp(ik[dot]v) for amplitude A and
	some vector v. The amplitudes A are given by the n-index (j,k,...,l)
	sum (over all possible jump types) of c(j,k,...,l)*p_j*r_j*r_k*...*r_l,
	where r_j is the hopping rate for the jump type j, p_j is the equilibrium
	occupation probability for sites with jump j, and c(...) is some
	coefficient. This function returns these coefficients.

	All possible types of jump are identified by unique hashable objects.
	These should distinguish every possible combination of the displacement
	vector and the hopping rate for a jump.

	The return value is a dictionary of dictionaries. The keys of the
	outer dictionary are the vectors, v, for each component as tuples of
	cartesian components. The subdictionaries take tuples of jump
	identifiers as keys, with values given by the corresponding value of
	the coefficient c for the component with vector v.

	Since r_j*r_k is the same as r_k*r_j, the subdictionaries only contain
	each unique combination of jump type identifiers once, sorted in
	lexicographic order. The exception to this is the first jump in the
	sequence, which needs an extra factor of the equilibrium occupation
	probability, so the reordering excludes the first jump.
	
	Where the value of the correponding coefficient is zero, the key will
	(with rare exception) be omitted altogether.

	Arguments:
	  n                 The order of the derivative for which to return
	                    the coefficients
	  jump_exists_probs A dictionary of lattice structure probabilities.
	                    The keys should be tuples of 2-tuples, which
	                    specify jump sequences. The first element of the
	                    nth 2-tuple indicates whether the vector of the
	                    nth jump in the sequence displaces the rest of
	                    sequence , while the second element is a jump
	                    identifier object which specifies the type
	                    of the nth jump in the sequence. The value
	                    corresponding to a thus specified sequence is the
	                    probability that a randomly selected lattice site
	                    is a valid start point for that sequence.
	                    A KeyError will be raised if this dictionary does
	                    not contain entries for every possible key with
	                    length n.
	  jump_vects        A dictionary such that jump_vects[obj] is the
	                    displacement vector of the jump identified by the
	                    object 'obj' as a 2-component cartesian array-like
	  vect_eq_rtol      Specifies the error tolerance when comparing two
	                    vectors for equality. This accounts for floating
	                    point errors when two different combinations of
	                    jump vectors have the same end-end vector.
	                    Specifically, any vectors which differ by a
	                    distance of less than this multiplied by the
	                    longest sum of n vectors in jump_vects are
	                    treated as the same.
	"""
	# The precise form of these coefficients is documented better elsewhere,
	# but in short: each possible sequence of jump vectors with length n
	# gives an additive contribution to the component with a vector equal
	# to the sequence's end-end vector. This contribution is given by some
	# function of the kinetic parameters for the jumps in the sequence
	# (with which this function does not concern itself), multiplied by
	# the probability that a random lattice site is a valid start for the
	# sequence, multiplied by a sign that alternates with the number of
	# jumps not contributing to the end-end vector.
	output_dict = {}
	jump_types = sorted(jump_vects.keys())
	poss_jumps = (
		tuple((False,i) for i in jump_types)
		+ tuple((True,i) for i in jump_types)
	)
	# By normalisation, the 0th derivative is just 1
	if n == 0:
		return {(0,0) : {() : 1}}
	# Calculate an absolute vector tolerance using the maximum length of
	# a sum of n vectors
	max_vect_sqr_len = max(np.square(v).sum() for v in jump_vects.values())
	vect_eq_atol = n * vect_eq_rtol * np.sqrt(max_vect_sqr_len)
	# Run through all possible jump sequences of length n, incrementing
	# output_dict with the corresponding coefficients
	for seq in itertools.product(poss_jumps, repeat=n):
		# First determine the vector for this component as a tuple,
		# matching it with any existing keys in output_dict which are
		# within tolerance, and use this to obtain the corresponding
		# subdictionary of output_dict (empty if this is a new vector)
		seq_vec = np.sum(
			[(0,0)] + [jump_vects[i[1]] for i in seq if i[0]],
			# (the (0,0) term is to ensure the output has shape (2,))
			axis=0
		)
		for v in output_dict.keys():
			if np.square(np.array(v)-seq_vec).sum() < vect_eq_atol**2:
				seq_vec = v
				break
		seq_vec = tuple(seq_vec)
		if seq_vec not in output_dict:
			output_dict[seq_vec] = {}
		coef_dict = output_dict[seq_vec]
		# Determine the probability of the sequence
		seq_prob = jump_exists_probs[seq]
		# If the probability is zero the coefficient will definitely vanish,
		# so there's no point including it
		if seq_prob == 0:
			continue
		# The sign of the terms alternates with the number of jumps with
		# first element False in their 2-tuple
		sign = 1 if sum(1 for i in seq if not i[0]) % 2 == 0 else -1
		# Increment the appropriate coefficient in coef_dict.
		# The key therein is a tuple of the jump type identifiers, with
		# all but the first one sorted in lexicographic order.
		coeff = sign * seq_prob
		key = (seq[0][1],) + tuple(sorted(i[1] for i in seq[1:]))
		if key in coef_dict:
			coef_dict[key] += coeff
		else:
			coef_dict[key] = coeff
	return output_dict


def ISF_init_deriv_components(
	deriv_order, jump_exists_probs, jump_vects,
	hop_rates, rel_eq_occptns, hop_rate_coeffs=None
):
	"""
	Returns the predicted complex exponential components of the value of
	the deriv_order-th time derivative of the ISF evaluated at t=0.

	The value of the nth derivative at t=0 is a linear combination of complex
	exponential components of the form A*exp(ik[dot]v) for amplitude A and
	some vector v. This function returns a dictionary with the vectors v
	(as tuples of cartesian components) for keys and the amplitudes A for
	values.

	All possible types of jump are identified by unique hashable objects.
	These should distinguish every possible combination of the displacement
	vector and the hopping rate for a jump.

	Arguments:
	  deriv_order       The order of the derivative for which to return
	                    the coefficients
	  jump_exists_probs A dictionary with 2-tuples for keys and values. The
	                    first element of each key should be a jump identifier
	                    and the second element should be a tuple of such
	                    identifiers. The first element of the value should
	                    be the probability that a site selected uniformly
	                    at random from all those points from which the
	                    sequence of jumps in the key's second element exists
	                    also has a jump of the type identified in the
	                    key's first element leading TO it. The second
	                    value is the probability that such a point has
	                    a jump of that type away FROM it.
	                    A KeyError will be raised if this dictionary does
	                    not contain entries for every possible key with
	                    a second element of length in [0, n-1].
	  jump_vects        A dictionary such that jump_vects[obj] is the
	                    displacement vector of the jump identified by the
	                    object 'obj' as a 2-component cartesian array-like
	  hop_rates         A dictionary such that hop_rates[obj] is the
	                    hopping rate (hops per time unit) for the jump
	                    identified by the object 'obj'
	  rel_eq_occptns    A dictionary such that rel_eq_occptns[obj] is the
	                    relative equilibrium occupation probability (as
	                    defined in section 3.2 of the report) of the
	                    type of site from which the jump identified by the
	                    object 'obj' originates
	  hop_rate_coeffs   An optional precalculated set of coefficients given
	                    by the ISF_init_deriv_hop_rate_coeffs() method,
	                    which can be reused for efficiency
	"""
	# Get the coefficients in the amplitudes for the hop rates
	if hop_rate_coeffs is not None:
		comp_coef_dict = hop_rate_coeffs
	else:
		comp_coef_dict = ISF_init_deriv_hop_rate_coeffs(
			deriv_order, jump_exists_probs, jump_vects
		)
	# Run through each component, perform the sum over the hopping rates
	# and update output_dict accordingly
	output_dict = {}
	for comp_vect in comp_coef_dict:
		coef_dict = comp_coef_dict[comp_vect]
		# The keys of coef_dict are tuples of the form (j,k,...,l) and the
		# amplitude we need is given by the sum over these indices of the
		# corresponding value multiplied by p_j*(r_j*r_k*...*r_l) (where
		# r_j is the hop rate for jump type j and p_j is the equilibrium
		# occupation probability)
		output_dict[comp_vect] = sum(
			coef_dict[key] * rel_eq_occptns[key[0]] * np.prod([hop_rates[i] for i in key])
			for key in coef_dict
		)
	return output_dict


def calc_jump_exists_probs(
	latt, possible_jump_types, max_num_jumps, categorise_jump,
	print_progress=False, print_raw_counts=False, seq_starts=None
):
	"""
	Calculate and return the jump_exists_probs argument corresponding to
	the specified jump sequences on a lattice.Lattice in the form required
	for predicted_ISF() and related functions.

	Probabilities are calculated by counting the occurances of each jump
	sequence on the provided lattice.

	Jumps should be identified by a set of arbitrary hashable objects.
	An identifier should uniquely define both the jump's displacement
	vector and any relevant categorisation.

	Arguments:
	  latt                  The lattice for which to calculate the
	                        probabilities.
	  possible_jump_types   A list of identifiers for all jump types
	                        which should be included in the count.
	  max_num_jumps         The maximum length of a sequence for which to
	                        calculate probabilities.
	  categorise_jump       A callable which defines the identification
	                        of jumps. It should take 2 lattice.LatticePoint
	                        objects as its arguments and return the
	                        identifier for the type of the jump from the
	                        first to the second (or None if there is no
	                        valid jump between them).
	  print_progress        Bool whether to print the current progress
	                        percentage to stdout while running.
	  print_raw_counts      Whether to print to stdout the raw counts of
	                        the number of occurances on the lattice before
	                        returning. Mostly for debugging purposes, but
	                        if any of the values are small but non-zero,
	                        the resulting probabilities are likely subject
	                        to significant error.
	  seq_starts            A list of tuples of jump identifiers (two-tuples
	                        as in the output, not just the type on its own)
	                        containing the possible sequence starts. If
	                        specified, probabilities will only be returned for
	                        all possible sequences which start with the sequence
	                        of types contained in one of the elements of
	                        seq_starts. Sequences shorter than these will
	                        not be included
	"""
	# Handle this special case so it doesn't cause errors later
	if seq_starts is not None and len(seq_starts) == 0:
		return {}
	# The probabilities we need require counts for every combination of
	# up to max_num_jumps jumps, and for each of those every combination
	# of advancing to the jump's destination point or not for the next
	# point in the sequence (no overline or overline in the algebraic
	# notation).
	# Therefore, for each identifier object i returned by categorise_jump, 
	# our get_jumps function needs to return the identifier (False, i)
	# associated with the point in its argument and the identifier
	# (True, i) associated with that jump's desination.
	def get_jumps(p):
		return_list = []
		jump_types_to_p = []
		for q in p.nrst_neighbours:
			jump_type = categorise_jump(p,q)
			if jump_type is not None:
				return_list.append((q,[(True, jump_type)]))
				jump_types_to_p.append((False, jump_type))
		return_list.append((p, jump_types_to_p))
		return return_list
	# Likewise, get a complete list of all possible 2-tuple identifiers
	possible_jump_types = tuple(possible_jump_types)
	possible_jump_ids = (
		tuple((True, i) for i in possible_jump_types)
		+ tuple((False, i) for i in possible_jump_types)
	)
	# Generate every possible sequence of jump identifiers up to the
	# specified length
	jump_seqs = [()]
	if seq_starts is None:
		for seq_length in range(1, max_num_jumps+1):
			jump_seqs.extend(
				itertools.product(possible_jump_ids, repeat=seq_length)
			)
	else:
		for seq_start in seq_starts:
			for seq_length in range(0, 1+max_num_jumps-len(seq_start)):
				if seq_length == 0:
					jump_seqs.append(seq_start)
				else:
					jump_seqs.extend(
						seq_start + rest_of_seq
						for rest_of_seq in itertools.product(
							*((possible_jump_ids,) * (seq_length))
						)
					)
	# Perform the count on the lattice
	counts = latt.count_jump_sequences(jump_seqs, get_jumps, print_progress)
	if print_raw_counts: print(counts)
	# Construct the jump_exists_probs dictionary and populate it with the
	# probabilities of the sequences (ie their counts divided by the number
	# of zero-length sequences)
	jump_exists_probs = {}
	denom = counts[()]
	for count_id_seq in jump_seqs:
		# Exclude the zero length sequence if it wasn't requested
		if (
			count_id_seq != ()
			or seq_starts is None
			or min(len(s) for s in seq_starts) == 0
		):
			jump_exists_probs[count_id_seq] = counts[count_id_seq] / denom
	return jump_exists_probs


def identify_jump_vect(vects, position_tol):
	"""
	Returns a callable which identifies which displacement vector
	corresponds to the jump from one point to another.

	Specifically, the returned callable has signature func(start, end)
	(where start and end are LatticePoint objects) and returns the index
	of the first vector in the list vects which matches the difference
	in positions (end_pos - start_pos) of the two points to within a
	distance of approximately position_tol (distance measurement is only
	accurate to within about a factor of 4 for performance reasons). It
	returns None if none of the vectors match.

	Arguments:
	  vects             A list of 2-element array-likes containing the
	                    cartesian components of the possible displacement
	                    vectors
	  position_tol      The maximum magnitude of the difference between
	                    the separation of the points and a matching vector
	                    (to allow for floating point errors).
	"""
	# For performance reasons, create a dictionary with keys given by
	# the vectors rounded to a grid of spacing 2*position_tol (as tuples).
	# Specifically, make it so that dividing the components of a vector
	# by position_tol and rounding down to the nearest integer gives
	# a key which yields a value with the index of a vector which is within
	# at most about four times position_tol.
	# The value is a list of (vector, index) tuples, where index is the
	# index in vects.
	grid_mult_fact = .4999 / position_tol
	# (slightly larger grid in case of floating point errors)
	grid_dict = {}
	for i, vect in enumerate(vects[::-1]):
		# Make vect the value for every grid point whose corresponding
		# grid square intersects a circle of radius position_tol around v
		grid_points = set()
		for delta_v in position_tol * np.array(((1,0), (0,1), (-1,0), (0,-1))):
			test_point = np.array(vect) + delta_v
			rounded_tuple = tuple(np.trunc(grid_mult_fact * test_point))
			grid_points.add(rounded_tuple)
		for gp in grid_points:
			grid_dict[gp] = i
	# Return the appropriate callable
	def func(start, end):
		pos_delta = end.position - start.position
		try:
			return grid_dict[tuple(np.trunc(grid_mult_fact * pos_delta))]
		except KeyError:
			return None
	return func


def evenly_spaced_vectors(n, a, rotate_angle=0):
	"""
	Convenience function to calculate n evenly spaced vectors of length a.

	Returns a tuple of 2-element numpy arrays contaning the cartesian
	components of the n vectors obtained by rotating a vector of length
	a directed along the x-axis by multiples of 2pi/n.

	Optionally, all vectors are rotated anticlockwise by rotate_angle
	radians.
	"""
	angles = (2*np.pi*m/n + rotate_angle for m in range(n))
	return tuple(a * np.array([np.cos(phi), np.sin(phi)]) for phi in angles)


def get_matching_vect(vectors, vect_to_match, tol=1e-9):
	"""
	Return the first vector in the iterable of 2-element array-likes
	vect_to_match which is equal to vect_to_match within approximately
	a distance tol
	
	Returns None if one isn't found
	"""
	for v in vectors:
		if np.square(np.array(v)-vect_to_match).sum() < tol**2:
			return v
	return None
