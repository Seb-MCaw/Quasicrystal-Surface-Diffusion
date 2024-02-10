"""
Defines the types LatticePoint and Lattice to represent a general (finite)
lattice with arbitrary geometry, and subclasses to define specific
geometries.
"""


import itertools

import numpy as np
import matplotlib.pyplot as plt



class LatticePoint:
	"""
	Represents a single point on a lattice.

	Attributes:
	  position          A 1D numpy array of length 2 containing the lattice
						point's cartesion coordinates
	  nrst_neighbours   A list of LatticePoints which should be considered
	                    to be nearest neighbours to this one
	"""

	def __init__(self, position, nearest_neighbours=None):
		if nearest_neighbours is None: nearest_neighbours = []
		self.position, self.nrst_neighbours = position, nearest_neighbours

	def disconnect_from_neighbours(self):
		"""Severs all nearest neighbour relations with this lattice point."""
		for neighbour in self.nrst_neighbours:
			neighbour.nrst_neighbours.remove(self)
		self.nrst_neighbours = []

	def get_neighbour_at(self, relative_position, position_tol):
		"""
		Find a neighbouring lattice point at the specified relative position.

		Returns a(ny) neighbouring LatticePoint which is located within a
		distance position_tol from self.position + relative_position.
		If this point has no such neighbours, returns None.
		"""
		search_pos = self.position + np.array(relative_position)
		for nghbr in self.nrst_neighbours:
			if np.square(nghbr.position - search_pos).sum() < position_tol**2:
				return nghbr
		return None

	def jump_sequence_exists(self, jump_seq, get_jumps):
		"""
		Returns bool whether this point is a valid start for the specified
		sequence of jumps.

		A zero-length sequence of jumps always returns True.

		Jumps should be identified by a set of arbitrary (unique) hashable
		objects. An identifier should define both the jump's displacement
		vector and any relevant categorisation.

		Arguments:
		  jump_seq      A tuple-like of identifiers, defining the jump
		                sequence. This method returns true if self
		                possesses a jump of type jump_seq[0] from it, which
		                leads to a point with a jump of type jump_seq[1]
		                from it, which leads to a point with one of type
		                jump_seq[2] from it, etc.
		  get_jumps     A callable which defines the identification of
		                jumps. When a LatticePoint is passed as its sole
		                argument, it should return a dictionary with keys
		                given by the appropriate identifiers for that
		                point's jumps, and the corresponding values given
		                by the points to which those jumps lead.
		"""
		jump_seq = tuple(jump_seq)
		if len(jump_seq) == 0:
			return True
		jumps_from_self = get_jumps(self)
		if jump_seq[0] in jumps_from_self:
			next_point = jumps_from_self[jump_seq[0]]
			return next_point.jump_sequence_exists(jump_seq[1:], get_jumps)
		else:
			return False

	def increment_jump_sequence_counts(self, sequence_count_dict, get_jumps):
		"""
		Takes a set of counters for a number of different jump sequences and
		increments those for which self.jump_sequence_exists() is True (but
		does so more efficiently than repeated calls thereto).

		For performance reasons, the counters should be passed as nested
		dictionaries terminated by key values of None, ie such that the
		counter for the sequence a,b,c (where each of these is an object
		identifying a jump type) is sequence_count_dict[a][b][c][None] .
		sequence_count_dict[None] is always incremented (if present).

		Jumps should be identified by a set of arbitrary (unique) hashable
		objects. An identifier should define both the jump's displacement
		vector and any relevant categorisation.

		Jump categorisation is achieved by the callable argument get_jumps.
		When a LatticePoint p is passed as its sole argument, it should
		return a list of 2-tuples containing all of the jump types which
		exist from p, categorised by the point from which the rest of the
		sequence thereafter should be counted. The next points in the
		sequence are the first element of each 2-tuple and the second
		element should be a list of the corresponding jump types. Duplicate
		points will lead to incorrect counts.
		"""
		self._increment_multiple_jump_sequence_counts(
			[sequence_count_dict], get_jumps
		)

	def _increment_multiple_jump_sequence_counts(self, sequence_count_dicts, get_jumps):
		"""
		Equivalent to increment_jump_sequence_counts, but takes a list of
		multiple sequence_count_dict dictionaries, and increments all of them.
		"""
		jumps_from_self = list(get_jumps(self))
		# Make a copy of the list sequence_count_dicts (but not the
		# dictionaries therein), because we might need to modify it in
		# what follows
		sequence_count_dicts = sequence_count_dicts[:]
		# Compile a list of other points at which subdictionaries need to
		# be incremented by a recursive call to this method.
		# The elements of this list are tuples, with the point at which
		# to make the recursive call as the first element and a list of
		# the corresponding subdictionaries as the second
		sub_dicts_by_point = []
		for sequence_count_dict in sequence_count_dicts:
			if None in sequence_count_dict:
				sequence_count_dict[None] += 1
				if len(sequence_count_dict) == 1:
					# We don't need to count any further jumps
					continue
			for next_point, jump_type_list in jumps_from_self:
				jump_type_list = list(jump_type_list)
				sub_dicts_list = next(
					(t[1] for t in sub_dicts_by_point if t[0] is next_point),
					None
				)
				if sub_dicts_list is None:
					sub_dicts_list = []
					sub_dicts_by_point.append((next_point, sub_dicts_list))
				for jump_type in jump_type_list:
					try:
						sub_dict = sequence_count_dict[jump_type]
					except KeyError:
						pass
					else:
						# If next_point is still self, then we can just add
						# this subdictionary on to the list we're currently
						# processing, but otherwise we will need to save it
						# for a recursive call to this method later
						if next_point is self:
							sequence_count_dicts.append(sub_dict)
						else:
							sub_dicts_list.append(sub_dict)
		# Then go through the list of points and invoke this method
		# recursively on each with the appropriate list of dictionaries
		for next_point, sub_dicts in sub_dicts_by_point:
			if len(sub_dicts) > 0:
				next_point._increment_multiple_jump_sequence_counts(
					sub_dicts, get_jumps
				)


class Lattice:
	"""
	Represents a finite lattice of LatticePoint objects.

	Attributes:
	  points            A dictionary with instances of LatticePoint containing
	                    the points on the lattice for values, and tuples
						of their cartesian coordinates as keys. Points
						should be added with the add_point() method and
						removed with the remove_point() method.
	  defined_region    The range of cartesian coordinates over which the
	                    lattice is properly defined. Simulations which
						find themselves outside of this region should throw
						a RuntimeError. Its value is a callable with takes
						an array-like containing the cartesian coordinates
						of a point as an argument, and returns a boolean
						as to whether that point is within the properly
						defined region. 
	"""

	def __init__(self, _, defined_region=None, lattice_points=None):
		"""
		If defined_region is not specified, the lattice is taken as being
		properly defined everywhere (ie its finiteness is physical).
		
		The first instantiation argument is deprecated.
		"""
		if defined_region is None: defined_region = lambda _: True
		if lattice_points is None: lattice_points = {}
		self.points = lattice_points
		self.defined_region = defined_region
		# For speed when calling get_point_by_coords(), create a dictionary
		# with position tuples for keys and lists of lattice points for
		# values such that a every point located within a square on a
		# square grid can be found in the list corresponding to its position
		# rounded to the nearest grid point.
		self._grid_size_dec_places = 1
		self._points_rounded_to_grid = {}
		for p in lattice_points.values():
			gp = tuple(np.floor(10*p.position))
			if gp in self._points_rounded_to_grid:
				self._points_rounded_to_grid[gp].append(p)
			else:
				self._points_rounded_to_grid[gp] = [p]

	def __getitem__(self, pos):
		return self.get_point_by_coords(pos)

	def add_point(self, new_lattice_point):
		"""Adds the LatticePoint instance new_lattice_point to the lattice"""
		self.points[tuple(new_lattice_point.position)]
		gp = tuple(np.floor(10*new_lattice_point.position))
		if gp in self._points_rounded_to_grid:
			self._points_rounded_to_grid[gp].append(new_lattice_point)
		else:
			self._points_rounded_to_grid[gp] = [new_lattice_point]

	def remove_point(self, lattice_point_to_remove):
		"""Remove the lattice point and all references to it from the lattice"""
		# Remove references from nearest neighbours
		lattice_point_to_remove.disconnect_from_neighbours()
		# Remove from the Lattice object
		self.points.pop(tuple(lattice_point_to_remove.position))
		gp = tuple(np.floor(10*lattice_point_to_remove.position))
		self._points_rounded_to_grid[gp].remove(lattice_point_to_remove)

	def get_point_by_coords(self, pos):
		"""
		Returns the lattice point found at position pos, even if pos is
		subject to floating point errors.

		Returns the point as a LatticePoint if there is one, or None if not.
		If there are multiple points close to pos, the first one found is
		returned.

		This is much slower than calling Lattice.points[pos], so only use
		this if floating point errors are expected.
		"""
		grid_points = set(
			tuple(np.floor(10*pos) + np.array(d))
			for d in itertools.product((-2,-1,0,1,2), repeat=2)
		)
		for gp in grid_points:
			if gp in self._points_rounded_to_grid:
				for lp in self._points_rounded_to_grid[gp]:
					if max(np.abs(lp.position - np.array(pos))) < 1e-2:
						return lp
		return None

	def get_points_near(self, pos, dist):
		"""
		Return a list of all points within APPROXIMATELY a distance dist of
		the position pos
		"""
		grid_points = set(
			tuple(np.floor(10*np.array(pos)) + np.array(d))
			for d in itertools.product(range(-int(10*dist)-2, int(10*dist)+3), repeat=2)
		)
		return [
			lp
			for gp in grid_points
			if gp in self._points_rounded_to_grid
			for lp in self._points_rounded_to_grid[gp]
			if max(np.abs(lp.position - np.array(pos))) <= dist
		]

	def is_in_defined_region(self, position):
		"""
		Convenience method to returns bool whether position (an array-like
		of cartesian coordinates) is within the lattice's defined region
		"""
		return self.defined_region(position)

	def link_points_separated_by(self, lattice_vectors):
		"""
		Creates nearest-neighbour relations between any pairs of lattice
		points that are separated by a member (or the negative of a
		member) of lattice_vectors and are not already related.

		lattice_vectors is an array-like of 2-element array-likes, each
		of which contains the cartesian coordinates of the vector.
		"""
		# Flip lattice vectors (sign is irrelevant) as required to ensure
		# they are along a polar angle phi in the range (-pi/2, pi/2]
		# Also take the opportunity to convert them into tuples
		lattice_vectors = [
			((-v[0], -v[1]) if (v[0]>0 or (v[0]==0 and v[1]>0)) else (v[0], v[1]))
			for v in lattice_vectors
		]
		# Remove any redundant vectors (ordering is irrelevant)
		lattice_vectors = set(lattice_vectors)
		# Loop through all lattice points and add relations where there
		# is a point in the right relative position
		for lv in lattice_vectors:
			lat_vect = np.array(lv)
			for this_point in self.points.values():
				nrst_nghbr_pos = this_point.position + lat_vect
				other_point = self.get_point_by_coords(nrst_nghbr_pos)
				if (
					other_point is not None
					and not other_point in this_point.nrst_neighbours
				):
					this_point.nrst_neighbours.append(other_point)
					other_point.nrst_neighbours.append(this_point)

	def count_jump_sequences(self, jump_sequences, get_jumps, print_progress=False):
		"""
		Count the number of points in the lattice from which each of the
		specified sequences of jumps are valid sets of hops between
		neighbouring lattice points.

		Returns a dictionary where the keys are the sequences in
		jump_sequences and the values are integers containing a count of
		the number of points from which they exist.

		Jumps should be identified by a set of arbitrary hashable objects.
		An identifier should uniquely define both the jump's displacement
		vector and any relevant categorisation.

		Arguments:
		  jump_sequences    A list of jump sequences. Each is defined by
		                    a tuple of identifier objects corresponding
		                    to each jump.
		  get_jumps         A callable which defines the identification of
		                    jumps. When a LatticePoint p is passed as its
		                    sole argument, it should return a list of
		                    2-tuples containing all of the jump types which
		                    exist from p, categorised by the point from
		                    which the rest of the sequence thereafter
		                    should be counted. The next points in the
		                    sequence are the first element of each 2-tuple
		                    and the second element should be a list of
		                    the corresponding jump types. Duplicate points
		                    will lead to incorrect counts.
		  print_progress    Bool whether to print the current progress
		                    percentage to stdout while running.
		"""
		jump_sequences = list(jump_sequences)
		# Generate the counter dictionary in the format required by
		# LatticePoint.increment_jump_sequence_counts(), with all counts
		# initialised to zero.
		count_dict = {}
		for sequence in jump_sequences:
			sub_dict = count_dict
			for jump_type in sequence:
				if jump_type not in sub_dict:
					sub_dict[jump_type] = {}
				sub_dict = sub_dict[jump_type]
			sub_dict[None] = 0
		# Run through every point in the lattice and increment the counters
		# for each valid sequence
		for i, pnt in enumerate(self.points.values()):
			if print_progress and i % 100 == 0:
				print(f"\r{100*i/len(self.points):.2f}%", end="")
			pnt.increment_jump_sequence_counts(count_dict, get_jumps)
		if print_progress:
			print("\r100.00%")
		# Transform count_dict into the desired format and return it
		out_dict = {}
		for sequence in jump_sequences:
			sub_dict = count_dict
			for jump_type in sequence:
				sub_dict = sub_dict[jump_type]
			out_dict[sequence] = sub_dict[None]
		return out_dict

	def count_all_jump_sequences(
		self, possible_jump_types, max_num_jumps, get_jumps, print_progress=False
	):
		"""
		Count the number of points in the lattice from which each possible
		combination of up to max_num_jumps chosen from possible_jump_types
		is a valid sequence of jumps between neighbouring lattice points.

		Jumps should be identified by a set of arbitrary hashable objects.
		An identifier should uniquely define both the jump's displacement
		vector and any relevant categorisation.

		Returns a dictionary where the keys are tuples containing sequences
		of jump type identifier objects corresponding to the jump sequence
		and the values are integers containing the count of the number of
		points starting from which this is a valid sequence.

		Arguments:
		  possible_jump_types   A list of identifiers for the jump types
		                        which should be included in the count.
		  max_num_jumps         The maximum length of a sequence to be
		                        included.
		  get_jumps             A callable which defines the identification of
		                        jumps. When a LatticePoint p is passed as its
		                        sole argument, it should return a list of
		                        2-tuples containing all of the jump types which
		                        exist from p, categorised by the point from
		                        which the rest of the sequence thereafter
		                        should be counted. The next points in the
		                        sequence are the first element of each 2-tuple
		                        and the second element should be a list of
		                        the corresponding jump types. Duplicate	points
		                        will lead to incorrect counts.
		  print_progress        Bool whether to print the current progress
		                        percentage to stdout while running.
		"""
		possible_jump_types = list(possible_jump_types)
		# Generate a list of all possible jump sequences up to the
		# specified length
		num_types = len(possible_jump_types)
		jump_seqs = []
		for seq_length in range(max_num_jumps+1):
			for seq in itertools.product(range(num_types), repeat=seq_length):
				jump_seqs.append(tuple(possible_jump_types[i] for i in seq))
		# Perform count and return result
		return self.count_jump_sequences(
			jump_seqs, get_jumps, print_progress
		)

	def show_plot(self, show_points=False):
		"""
		Displays a matplotlib popup with a plot of the lattice.
		
		The lattice is represented by displaying nearest neighbour
		relationships as lines.
		"""
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.set_xlabel(r"x")
		ax.set_ylabel(r"y")
		for lattice_point in self.points.values():
			# Plot a line for every nearest neighbour which has a larger
			# x coordinate or has the same x coordinate and a larger y
			# coordinate (to avoid doubling lines unnecessarily)
			for nrst_nghbr in lattice_point.nrst_neighbours:
				if ((nrst_nghbr.position[0] > lattice_point.position[0])
						or (nrst_nghbr.position[0] == lattice_point.position[0]
						    and nrst_nghbr.position[1] > lattice_point.position[1])):
					ax.plot(
						[lattice_point.position[0], nrst_nghbr.position[0]],
						[lattice_point.position[1], nrst_nghbr.position[1]],
						linestyle="-",
						color="black")
		if show_points:
			x, y = zip(*[tuple(lp.position) for lp in self.points.values()])
			ax.scatter(x, y)
		plt.show()

	def fourier_transform(
		self, k_vectors, print_progress=False, filter=None, gaussn_wndw_sd=None
	):
		"""
		Returns an array containing the value of the Fourier Transform
		of the lattice at the specified k values.

		In other words, this calculates the sum over all the points in
		the lattice of e^(-ik [dot] r), where r is the position of the
		point.

		Arguments:
			k_vectors       An array(-like) of 2-element array(-like)s.
			                ie an array with length 2 in the 1 dimension,
			                such that delta_k_vectors[i][0] contains the
			                x component of the ith delta_k value.
			print_progress  Whether to print progress to stdout as a
			                percentage.
			filter          A callable which, when passed lattice point,
			                should return bool whether the point should
			                be included in the calculation of the Fourier
			                transform
			gaussn_wndw_sd  The standard deviation in real space of the
			                Gaussian window function (no window function
			                is used if unspecified)
		"""
		k_x, k_y = np.array(k_vectors).T
		FT_vals = 0
		for i, point in enumerate(self.points.values()):
			if filter is not None and not filter(point):
				continue
			if print_progress and i % int(1 + len(self.points)/10000) == 0:
				print(f"\r{100*i/len(self.points):.2f}%", end="")
			r_x, r_y = point.position
			if gaussn_wndw_sd is None:
				FT_vals = FT_vals + np.exp(-1j * (k_x*r_x + k_y*r_y))
			else:
				FT_vals = FT_vals + np.exp(
					-1j * (k_x*r_x+k_y*r_y) - (r_x**2+r_y**2) / (2*gaussn_wndw_sd**2)
				)
		if print_progress: print("\r100.00%")
		return FT_vals

	def show_fourier_transform_intensity_in_2D(
		self, max_k, resolution, log_scale=True, intensity_lims=None,
		print_progress=False, filter=None, gaussn_wndw_sd=None
	):
		"""
		Plots the mod-squared of the Fourier transform of the lattice.
		"""
		# Create array of k values
		pixel_size = 2*max_k / (resolution-1)
		k_grid = np.mgrid[-max_k : max_k+pixel_size/2 : pixel_size,
		                  -max_k : max_k+pixel_size/2 : pixel_size].T
		# Calculate transform
		if print_progress: print("Calculating Fourier transform:")
		FT_vals = self.fourier_transform(
			k_grid.reshape(-1,2), print_progress, filter, gaussn_wndw_sd
		)
		# Reshape transform back into grid
		FT_vals = FT_vals.reshape(resolution, resolution)
		# Convert into plottable form
		FT_vals = np.abs(FT_vals)**2
		if log_scale: FT_vals = np.log(FT_vals)
		# Create plot
		if print_progress: print("Creating Plot...")
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.set_xlabel(r"$k_x$")
		ax.set_ylabel(r"$k_y$")
		vmin, vmax = (None, None) if intensity_lims is None else intensity_lims
		ax.imshow(FT_vals,
		          extent=[-max_k, max_k, -max_k, max_k],
		          cmap='gray',
		          vmin=vmin,
		          vmax=vmax)
		plt.show()


class ParallelogrammicLattice(Lattice):
	"""
	A finite 2 dimensional lattice with a primitive parrollelogrammic unit cell.

	All 2D Bravais lattices are special cases of this lattice.

	Has the attributes:
	  a                 The length of the side of the unit cell parallel
	                    to the x-axis
	  b                 The length of the other side of the unit cell (ie
	                    parallel to the y-axis)
	  theta             Angle (in radians) between the two unit cell sides
	                    (specifically anticlockwise from a to b)
	  perp_height       The perpendicular (to a) height of the unit cell
	  b_cos_theta       The (exact) value of b.cos(theta), which corresponds
	                    to the displacement in the x direction of the top
	                    (ie +ve y side) of the parallelogram relative to
	                    the bottom. Defined to avoid errors and computational
	                    overhead of working with trigonometric functions.
	"""

	def __init__(self, size_in_hops, a, perp_height, b_cos_theta):
		"""
		The size of the lattice generated is specified by size_in_hops,
		which is the maxmimum number of lattice-point to lattice-point
		hops which can be performed (starting from the origin) without
		reaching the edge of the defined area.

		The other instantiation arguments are the same as attributes.
		"""
		# Define unit cell attributes
		self.a, self.perp_height, self.b_cos_theta = a, perp_height, b_cos_theta
		self.b = np.sqrt(perp_height**2 + b_cos_theta**2)
		self.theta = np.arctan2(perp_height, b_cos_theta)
		# The lattice is defined within the area accessible by adding any
		# combination of size_in_hops lattice vectors
		defined_region = (lambda c:
			(
				abs(c[0] - c[1]*b_cos_theta/perp_height) / a
				+ abs(c[1]) / perp_height
			)
			<= size_in_hops
		)
		# Generate lattice points for every possible combination of up to
		# size_in_hops+1 unit cell vectors
		points = {}
		for num_a in range(-size_in_hops-1, size_in_hops+2):
			for num_b in range(abs(num_a)-size_in_hops-1, size_in_hops+2-abs(num_a)):
				x, y = num_a*a + num_b*b_cos_theta, num_b*perp_height
				points[x,y] = LatticePoint(np.array([x,y]))
		# Initialise object as a Lattice
		super().__init__(2, defined_region, points)
		# Add nearest neighbour relations
		self.link_points_separated_by([(a,0), (b_cos_theta,perp_height)])


class SquareLattice(ParallelogrammicLattice):
	"""
	A finite 2 dimensional lattice with a primitive square unit cell.

	Corresponds to a parallelogrammic lattice with theta=90degrees and
	a=b=lattice_parameter.

	Has the attributes:
	  lattice_parameter  The side length of the square unit cell
	as well as those inherited from ParallelogrammicLattice.
	"""

	def __init__(self, size_in_hops, lattice_parameter):
		super().__init__(size_in_hops, lattice_parameter, lattice_parameter, 0)


class TriangularLattice(ParallelogrammicLattice):
	"""
	A finite 2 dimensional hexagonal lattice (ie a triangular tiling).

	This has a rhombic unit cell with an obtuse interior angle of 120, so
	is the same as a ParallelogrammicLattice with a=b and theta=120degrees
	except that the short diagonal of the unit cell is also considered a
	nearest neighbour relation. This additional hopping direction requires
	a larger area to be defined, but (for simplicity) this is currently
	achieved by generating the rhombic lattice over twice the area.

	Has the same attributes as ParallelogrammicLattice (of which only a
	is relevant).
	"""

	def __init__(self, size_in_hops, a):
		# Create a rhombic lattice (of twice the size)
		super().__init__(2*size_in_hops, a, a*np.sqrt(3)/2, a*.5)
		# Add the extra nearest-neighbour relations
		self.link_points_separated_by([[a*.5, -a*np.sqrt(3)/2]])


class HoneycombLattice(TriangularLattice):
	"""
	A finite 2 dimensional honeycomb (hexagonal tiling).

	This is the same as TriangularLattice, except with a third of the
	lattice points removed to leave a tiling of primitive hexagons.
	Specifically, the points are removed such that there is a point at
	the origin but not one at the position (-a, 0). This gives vertical
	columns of hexagons.

	Has the same attributes as ParallelogrammicLattice (of which only a
	is relevant).
	"""

	def __init__(self, size_in_hops, a):
		# Generate triangular lattice
		super().__init__(size_in_hops, a)
		# Specify a point to be removed to define location of lattice
		start_point = (-a, 0)
		# Run through all points that are displaced from this position by
		# some combination of the vectors (1.5a, perp_height) and (0, 2perp_height).
		# We can ignore points which don't exist, so doesn't bother with
		# carefully calculating where the defined region is.
		for num_y_vects in range(-4-2*size_in_hops, 4+2*size_in_hops):
			for num_other_vects in range(-4-2*size_in_hops, 4+2*size_in_hops):
				x = start_point[0] + 1.5*a*num_other_vects
				y = start_point[1] + self.perp_height*(2*num_y_vects+num_other_vects)
				point_to_remove = self.get_point_by_coords((x,y))
				if point_to_remove is not None:
					self.remove_point(point_to_remove)
