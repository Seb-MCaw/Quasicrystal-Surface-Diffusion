"""
Implements deflation-based generation of finite sections of a number of specific
2D Penrose tilings.
"""


import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import golden as GOLDEN_RATIO

import lattice


def new_tiling(radius, pattern="P2", tile_length=1, origin_shift=(0,0), defl_region_size=None):
	"""
	Generates and returns a Tiling object as specified.

	The tilings are generated such that there is a vertex at (0,0).

	Arguments:
	  radius            The distance from the origin within which the
	                    tiling is fully generated. More precisely, all
	                    vertices that are less than radius from the origin
	                    will be present and will be connected to the
	                    correct neighbouring vertices (some of which may
	                    lie outside the radius)
	  pattern           Specifies which tiling to use. See below for more
	                    information on which are currently supported.
	  tile_length       A length which defines the size of each tile. The
	                    exact length to which this refers depends on the
	                    pattern (see below)
	  origin_shift      An array-like containing the x and y components
	                    of an approximate displacement to
	                    subtract from all vertices in the tiling (some
	                    rounding required to put a vertex at (0,0))
	  defl_region_size  The radius of the region over which to perform
	                    the deflation process. If unspecified, inferred
	                    from radius and origin_shift. If specified, must
	                    be greater than radius + magnitude of origin_shift
	
	Patterns:
	  "P2"  The second tiling discovered by Penrose (and the first with
	        only 2-tiles). Consists of 'kite' and 'dart' tiles. The length
	        of the shorter sides of the tiles is given by tile_length (the
	        other length being longer by a factor of the golden ratio)
	  "P3"  The third tiling discovered by Penrose. Consists two rhombi
	        with the same side lengths but different angles. All tiles
	        have side lengths equal to tile_length.
	  "APM" A decoration of the P1 tiling based on the five-fold surface
	        of AlPdMn. The length of the sides of the P1 tiles are given
	        by tile_length and the returned tiling contains only vertices.
	"""
	origin_shift = np.array(origin_shift)
	if defl_region_size is None:
		defl_region_size = norm(origin_shift) + radius
	elif defl_region_size < norm(origin_shift) + radius:
		raise ValueError("defl_region_size too small")
	# Create an initial tiling which is large enough to completely contain
	# a circle of the specified radius at the specified position and has
	# a tile size such that an integer number of deflation generations
	# will yield tiles with the correct tile_length. Then perform those
	# deflations, and then shift the origin as required
	if pattern == "P2":
		# The initial tiling consists of a single kite, positioned such
		# that the first deflation will put the central vertex at the origin
		#
		# First calculate some useful geometric values
		cos_36_degrees = GOLDEN_RATIO / 2
		sin_36_degrees = np.sqrt(1 - GOLDEN_RATIO**2 / 4)
		cos_72_degrees = cos_36_degrees**2 - sin_36_degrees**2
		sin_72_degrees = 2 * sin_36_degrees * cos_36_degrees
		# The minimum distance from the origin to the edge of the tiling
		# is the perpendicular distance to one of the shorter edges of
		# the large kite. This must be at least equal to defl_region_size
		# (add a 1% margin to account for any slight innacuracies in the
		# floating point arithmetic - the precise percentage is unimportant).
		min_initial_tile_length = 1.01 * defl_region_size / sin_72_degrees
		# Find the minimum power of phi greater than this (when
		# multiplied by tile_length)
		num_deflations = math.ceil(np.log(min_initial_tile_length / tile_length)
		                           / np.log(GOLDEN_RATIO))
		# Create the initial vertices accordingly
		initial_tile_length = tile_length * GOLDEN_RATIO**num_deflations
		top_vertex = TilingVertex(
			GOLDEN_RATIO * initial_tile_length * np.array([cos_72_degrees, sin_72_degrees]))
		bttm_vertex = TilingVertex(
			GOLDEN_RATIO * initial_tile_length * np.array([cos_72_degrees, -sin_72_degrees]))
		left_vertex = TilingVertex(
			initial_tile_length * np.array([-GOLDEN_RATIO, 0]),
			[top_vertex, bttm_vertex])
		right_vertex = TilingVertex(
			initial_tile_length * np.array([1, 0]),
			[top_vertex, bttm_vertex])
		top_vertex.adjacent_vertices = [left_vertex, right_vertex]
		bttm_vertex.adjacent_vertices = [left_vertex, right_vertex]
		# Create the 2 half-kites
		upper_half_kite = P2HalfKite(GOLDEN_RATIO * initial_tile_length,
		                             (left_vertex, right_vertex, top_vertex))
		lower_half_kite = P2HalfKite(GOLDEN_RATIO * initial_tile_length,
		                             (left_vertex, right_vertex, bttm_vertex))
		# Create the initial Tiling object
		# We need dummy values of defined_region so that the first
		# deflations happen at all (currently all vertices are outisde
		# the radius)
		tiling = Tiling(
			[upper_half_kite, lower_half_kite],
			[top_vertex, bttm_vertex, left_vertex, right_vertex],
			(lambda x: True))
		# Perform one generation of deflation to create what we defined
		# as the initial pattern
		tiling.deflate(1)
		# Now deflate repeatedly within a deflation region of a radius
		# given by twice the current tile length plus the target radius.
		# This should ensure that no tiles are discarded prematurely and
		# that there are tiles around the edge to account for the pattern
		# being shifted slightly to put a point at the origin.
		for defl_step_num in range(num_deflations, 0, -1):
			cur_tile_length = tile_length * GOLDEN_RATIO**(defl_step_num)
			tiling.defined_region = (
				lambda x: norm(x-origin_shift) < radius+4*cur_tile_length)
			tiling.deflate(1)
		# Find the vertex closest to origin_shift and shift the whole
		# tiling so that this vertex is at (0,0)
		closest_vertex = tiling.nearest_vertex_to(origin_shift)
		# (We need to make a copy of closest_vertex.position because its
		# value will be changed as part of the shift process)
		shift_vector = closest_vertex.position.copy()
		for vertex in tiling.vertices:
			vertex.position -= shift_vector
		# Now at last replace dummy value of defined_region with its true value.
		tiling.defined_region = (lambda x: np.square(x).sum() < radius**2)
		# The tiling was slightly oversized, so do one last cleanup
		# with deflate(0) (should also reduce the chances of unhandled
		# artefacts around the edge), before returning the final tiling.
		tiling.deflate(0)
		return tiling
	elif pattern == "P3":
		# Generate the initial tiling
		tiling, num_deflations = p3_initial_tiling(tile_length, defl_region_size)
		# Now deflate repeatedly within a deflation region of a radius
		# given by twice the current tile length plus the target radius.
		# This should ensure that no tiles are discarded prematurely and
		# that there are tiles around the edge to account for the pattern
		# being shifted slightly to put a point at the origin.
		for defl_step_num in range(num_deflations, 0, -1):
			cur_tile_length = tile_length * GOLDEN_RATIO**(defl_step_num)
			tiling.defined_region = (
				lambda x: norm(x-origin_shift) < radius+4*cur_tile_length)
			tiling.deflate(1)
		# Find the vertex closest to origin_shift and shift the whole
		# tiling so that this vertex is at (0,0)
		# (We need to make a copy of closest_vertex.position because its
		# value will be changed as part of the shift process)
		closest_vertex = tiling.nearest_vertex_to(origin_shift)
		shift_vector = closest_vertex.position.copy()
		for vertex in tiling.vertices:
			vertex.position -= shift_vector
		# Now at last replace dummy value of defined_region with its true value.
		tiling.defined_region = (lambda x: np.square(x).sum() < radius**2)
		# We've generated a slightly oversized tiling, so do one last
		# cleanup with deflate(0) (should also reduce the chances of
		# unhandled artefacts around the edge), before returning the final
		# tiling.
		tiling.deflate(0)
		return tiling
	elif pattern == "APM":
		phi = GOLDEN_RATIO
		# Start by generating the parent P3 tiling
		p1_side = tile_length
		p3_side = p1_side * (1 + 2*phi) / (2*np.cos(np.pi/10))
		p3_tiling = new_tiling(
			radius + 10*p1_side, "P3", p3_side, origin_shift, defl_region_size
		)
		# Now generate most of the points by decorating the P3 tiles, keeping
		# track of the points that might be centres of the ambiguous pentagons
		# which we will fill in randomly later
		points = set()
		possible_ambiguous_pent_centres = set()
		for tile in p3_tiling.tiles:
			r0, r1, r2 = (v.position for v in tile.vertices)
			edge1 = r1 - r0
			unit_edge1 = edge1 / np.sqrt(np.square(edge1).sum())
			edge1_perp = r2 - (r0 + r1) / 2
			u_edge1_perp = edge1_perp / np.sqrt(np.square(edge1_perp).sum())
			if type(tile) is P3HalfFatRhomb:
				# The points making up P1
				point_1 = (2-phi) * r0 + (phi-1) * r1
				point_2 = point_1 + p1_side * (
					np.sin(3*np.pi/10) * u_edge1_perp - np.cos(3*np.pi/10) * unit_edge1
				)
				point_3 = point_2 + p1_side * (
					- np.sin(np.pi/10) * u_edge1_perp - np.cos(np.pi/10) * unit_edge1
				)
				point_4 = point_1 + p1_side * (
					np.sin(np.pi/10) * u_edge1_perp + np.cos(np.pi/10) * unit_edge1
				)
				points.add(tuple(point_1))
				points.add(tuple(point_2))
				points.add(tuple(point_3))
				points.add(tuple(point_4))
				# Additional decoration points
				point_5 = point_1 + p1_side * (
					np.sin(3*np.pi/10) * u_edge1_perp + np.cos(3*np.pi/10) * unit_edge1
				)
				point_6 = point_1 + p1_side * (
					np.sin(5*np.pi/10) * u_edge1_perp + np.cos(5*np.pi/10) * unit_edge1
				)
				point_7 = point_1 + p1_side * (
					np.sin(9*np.pi/10) * u_edge1_perp + np.cos(9*np.pi/10) * unit_edge1
				)
				points.add(tuple(point_5))
				points.add(tuple(point_6))
				points.add(tuple(point_7))
				# Check if vertex 0 is a possible centre of an ambiguous pentagon
				# (this will include a few other possible cases as well)
				if len(tile.vertices[0].adjacent_vertices) == 5:
					possible_ambiguous_pent_centres.add(tuple(r0))
			else:
				# P1 Points
				point_1 = (phi/(1+2*phi)) * r2 + ((1+phi)/(1+2*phi)) * r0
				point_2 = point_1 + p1_side * (
					np.cos(3*np.pi/10) * unit_edge1 - np.sin(3*np.pi/10) * u_edge1_perp
				)
				point_3 = point_1 + p1_side * (
					np.cos(3*np.pi/10) * unit_edge1 + np.sin(3*np.pi/10) * u_edge1_perp
				)
				points.add(tuple(point_1))
				points.add(tuple(point_2))
				points.add(tuple(point_3))
		# Convert the parent P3 tiling to a lattice object and apply the
		# vertex decoration rules
		p3l = p3_tiling.to_lattice()
		for lp in p3l.points.values():
			edges = tuple(
				0
				if lp.get_neighbour_at(
					[p3_side*np.cos(n*np.pi/5), p3_side*np.sin(n*np.pi/5)],
					1e-3
				) is None
				else 1
				for n in range(10)
			)
			if sum(edges) == 3:
				for shift in range(10):
					if edges[shift:] + edges[:shift] == (1,0,0,0,1,0,1,0,0,0):
						pent_top = lp.position + (
							(p1_side / (2*np.sin(np.pi/5)))
							* np.array([np.cos(shift*np.pi/5),
							np.sin(shift*np.pi/5)])
						)
						points.add(tuple(
							pent_top - p1_side * np.array([
								np.cos((shift+.5) * np.pi/5),
								np.sin((shift + .5) * np.pi/5)
							])
						))
						points.add(tuple(
							pent_top - p1_side * np.array([
								np.cos((shift - .5) * np.pi/5),
								np.sin((shift - .5) * np.pi/5)
							])
						))
						break
			if sum(edges) == 4:
				for shift in range(10):
					if edges[shift:] + edges[:shift] == (0,0,1,0,1,0,1,0,1,0):
						pent_top = lp.position + (
							(p1_side / (2*np.sin(np.pi/5)))
							* np.array([np.cos(shift*np.pi/5),
							np.sin(shift*np.pi/5)])	
						)
						points.add(tuple(
							pent_top - p1_side * np.array([
								np.cos((shift + .5) * np.pi/5),
								np.sin((shift + .5) * np.pi/5)
							])
						))
						points.add(tuple(
							pent_top - p1_side * np.array([
								np.cos((shift - .5) * np.pi/5),
								np.sin((shift - .5) * np.pi/5)
							])
						))
						break
		# Likewise fill in any ambiguous pentagons with two points with random
		# orientation
		for lp in [p3l.points[pos] for pos in possible_ambiguous_pent_centres]:
			edges = tuple(
				0
				if lp.get_neighbour_at(
					[p3_side*np.cos(n*np.pi/5), p3_side*np.sin(n*np.pi/5)],
					1e-3
				) is None
				else 1 for n in range(10)
			)
			if edges == (1,0,1,0,1,0,1,0,1,0) or edges == (0,1,0,1,0,1,0,1,0,1):
				shift = edges[1]
				angle = shift*np.pi/5 + np.random.randint(5) * 2*np.pi/5
				pent_top = lp.position + (
					(p1_side / (2*np.sin(np.pi/5)))
					* np.array([np.cos(angle), np.sin(angle)])
				)
				points.add(tuple(
					pent_top - p1_side * np.array([
						np.cos(angle + np.pi/10),
						np.sin(angle + np.pi/10)
					])
				))
				points.add(tuple(
					pent_top - p1_side * np.array([
						np.cos(angle - np.pi/10),
						np.sin(angle - np.pi/10)
					])
				))
		# Create a new tiling object containing these points
		return Tiling(
			[],
			[
				TilingVertex(np.array(p))
				for p in points if np.square(p).sum() < (radius + 10*p1_side)**2
			],
			(lambda x: np.square(x).sum() < radius**2)
		)
	else:
		raise ValueError(f"The pattern '{pattern}' is not supported")


class TilingVertex:
	"""
	Represents a vertex on a tiling

	Attributes:
	  position          A 1D numpy array of length 2 containing the
	                    cartesion coordinates of the vertex
	  adjacent_vertices A list of TilingVertex objects which are connected
	                    to this one by an edge
	"""
	# Also has the private attribute self._this_as_lattice_point, which
	# is initially None, but can be initialised to an empty placeholder
	# instance of lattice.LatticePoint the first time self._placeholder_lp()
	# is called, which can then be used by self.to_lattice_point() until
	# it is reset

	def __init__(self, position, adjacent_vertices=None):
		if adjacent_vertices is None: adjacent_vertices = []
		self.position = position
		self.adjacent_vertices = adjacent_vertices
		self._this_as_lattice_point = None

	def distance_from(self, other_vertex):
		"""Returns the distance from this vertex to the other vertex"""
		return norm(self.position - other_vertex.position)

	def disconnect_all_edges(self):
		"""Removes all adjacencies of this vertex with others and vice versa"""
		for v in self.adjacent_vertices:
			v.adjacent_vertices.remove(self)
		self.adjacent_vertices = []

	def _placeholder_lp(self):
		if self._this_as_lattice_point is None:
			self._this_as_lattice_point = lattice.LatticePoint(self.position)
		return self._this_as_lattice_point

	def to_lattice_point(self):
		"""
		Returns an instance of lattice.LatticePoint equivalent to this vertex
		"""
		self._placeholder_lp() # Ensure placeholder has been initialised
		self._this_as_lattice_point.nrst_neighbours = [
			v._placeholder_lp() for v in self.adjacent_vertices
		]
		return self._this_as_lattice_point


class Tile:
	"""
	A template class for the tiles that make up a tiling

	In general, these will only be part of a full tile in the final tiling
	in order to implement the deflation process.

	Attributes:
	  vertices          A tuple of TilingVertex objects, representing
	                    the vertices of the tile. In general, the order
	                    of these is important.
	  length            A length defining the size of the tile. Exactly
	                    which dimension this refers to will depend on
						the type of tile.
	"""

	def __init__(self, length=1, vertices=()):
		self.vertices, self.length = vertices, length
	
	def disconnect_non_shared_edges(self):
		raise NotImplementedError

	def deflate_once(self):
		"""
		This should perform the deflation process appropriate to the type
		of tile and return two lists; the first should contain the new
		tiles with which to replace this one and the second should contain
		any new vertices created by the process.
		"""
		raise NotImplementedError(f"The class {type(self).__name__} is a "
		                          f"subclass of penrose_tilings.Tile but "
		                          f"lacks a deflate_once() method")


class Tiling:
	"""
	Represents a finite section of a penrose tiling.

	Attributes:
	  tiles             A list of Tile objects containing the tiles (or
	                    tile parts) that make up the tiling
	  vertices          A list of TilingVertex objects containing all of
	                    the vertices in the tiling
	  defined_region    The range of cartesian coordinates over which the
	                    tiling is properly defined. Its value is a
	                    callable with takes an array-like containing the
	                    cartesian coordinates of a point as an argument,
	                    and returns a boolean as to whether that point
	                    is within the properly defined region. 
	"""

	def __init__(self, tiles=None, vertices=None, defined_region=(lambda x: True)):
		if tiles is None: tiles = []
		if vertices is None: vertices = []
		self.tiles, self.vertices = tiles, vertices
		self.defined_region = defined_region

	def nearest_vertex_to(self, position):
		"""
		Returns the vertex of this tiling closest to the specified position.

		Potentially very inefficient for large tilings.
		"""
		position = np.array(position)
		closest_vertex = self.vertices[0]
		closest_dist = norm(self.vertices[0].position - position)
		for vertex in self.vertices:
			dist = norm(vertex.position - position)
			if dist < closest_dist:
				closest_dist = dist
				closest_vertex = vertex
		return closest_vertex

	def _deflate_once(self):
		"""
		Perform the deflation process once for every tile in the tiling.

		This method doesn't completely clean up after itself, so should
		only be used internally. To perform a single deflation,
		self.deflate(1) should be used instead.

		Tiles which lie entirely outside the defined region will be
		discarded but their vertices will be left unaltered
		"""
		# Old tiles will be discarded, so we start with a new empty list
		# for them, but vertices will mostly stay the same with a few
		# added, so we will append/extend the existing self.vertices
		new_tile_list = []
		for tile in self.tiles:
			# Skip tiles which lie entirely outside the defined region
			if any(self.defined_region(v.position) for v in tile.vertices):
				new_tiles, new_vertices = tile.deflate_once()
				new_tile_list.extend(new_tiles)
				self.vertices.extend(new_vertices)
		# Replace the old tile list with the new one
		self.tiles = new_tile_list

	def deflate(self, num_generations):
		"""
		Deflate the tiling the specified number of times.		

		Vertices which lie outside the defined region and which, after
		the deflation process, are only connected to vertices which also
		lie outside the region will be discarded. Likewise, tiles entirely
		outside the defined region will be discarded.
		"""
		# Perform the deflations
		for i in range(num_generations):
			self._deflate_once()
		# Tiles will have already been discarded, so just go through and
		# discard vertices as required
		new_vertices_list = []
		for vertex in self.vertices:
			if (self.defined_region(vertex.position)
					or any(self.defined_region(v.position)
					       for v in vertex.adjacent_vertices)):
				new_vertices_list.append(vertex)
			else:
				vertex.disconnect_all_edges()
		self.vertices = new_vertices_list

	def to_lattice(self):
		"""
		Returns an instance of lattice.Lattice equivalent to this tiling
		
		Vertices become lattice points, which are treated as nearest
		neighbours iff they share an edge.
		"""
		# Reset the _this_as_lattice_point attributes of all vertices.
		# This allows for the very unlikely event an equivalent lattice
		# is generated and then modified, and then this method is called
		# again.
		for v in self.vertices:
			v._this_as_lattice_point = None
		# Create the list of LatticePoint objects
		points = {tuple(v.position):v.to_lattice_point() for v in self.vertices}
		return lattice.Lattice(2, self.defined_region, points)

	def count_tiles(self):
		"""
		Returns a dictionary of the number of tiles in the tiling of each type

		The keys are type objects (subclasses of Tile, of course) and
		the values are integers
		"""
		count_dict = {}
		for tile in self.tiles:
			if type(tile) in count_dict:
				count_dict[type(tile)] += 1
			else:
				count_dict[type(tile)] = 1
		return count_dict


class P2HalfKite(Tile):
	"""
	A triangle given by one half of the Kite tile from the P2 pattern

	The edge described by the first two elements of the vertices tuple
	is the diagonal of the whole kite tile, so these vertices should not
	be connected by an edge. That described by the second and third should
	form the shorter external side and that described by the third and first
	should form the longer side of the overall kite. Their lengths
	relative to this object's length attribute should be phi, 1 and phi
	respectively.

	Deflation yields two P2HalfKites (forming a kite) and one P2HalfDart.
	"""

	# __init__() is inherited from the parent class

	def deflate_once(self):
		vertex_1, vertex_2, vertex_3 = self.vertices
		# Create a vertex along the first edge such that it is phi times
		# further from the first vertex than the second. The neighbouring
		# tile may have already been deflated, so we need to check if
		# such a vertex has already been created and use that if so.
		# For consistency with the case where it does already exist, a
		# new vertex must therefore contain references to the relevant
		# existing vertices on the shared edge.
		new_vertices = []
		new_vertex_pos = ((GOLDEN_RATIO*vertex_2.position + 1*vertex_1.position)
		                  / (1+GOLDEN_RATIO))
		first_new_vertex = TilingVertex(new_vertex_pos, [vertex_1, vertex_2])
		vertex_already_exists = False
		for v in vertex_1.adjacent_vertices:
			if v.distance_from(first_new_vertex) < self.length / 100:
				# All side lengths in the tiling are of order self.length,
				# so something within self.length/100 must be a duplicate
				# of the vertex we just created
				first_new_vertex = v
				vertex_already_exists = True
				assert v in vertex_2.adjacent_vertices
		if not vertex_already_exists:
			vertex_1.adjacent_vertices.append(first_new_vertex)
			vertex_2.adjacent_vertices.append(first_new_vertex)
			new_vertices.append(first_new_vertex)
		# Repeat for a vertex along the third edge, closer to the first
		# vertex than the third
		new_vertex_pos = ((GOLDEN_RATIO*vertex_1.position + 1*vertex_3.position)
		                  / (1+GOLDEN_RATIO))
		second_new_vertex = TilingVertex(new_vertex_pos, [vertex_3])
		vertex_already_exists = False
		for v in vertex_3.adjacent_vertices:
			if v.distance_from(second_new_vertex) < self.length / 100:
				second_new_vertex = v
				vertex_already_exists = True
		if not vertex_already_exists:
			vertex_3.adjacent_vertices.append(second_new_vertex)
			new_vertices.append(second_new_vertex)
			# Disconnect the 1st and 3rd vertices because there is now a
			# vertex on the edge between them
			vertex_1.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_1)
		# Connect the two new vertices together
		first_new_vertex.adjacent_vertices.append(second_new_vertex)
		second_new_vertex.adjacent_vertices.append(first_new_vertex)
		# Create the new half dart
		half_dart = P2HalfDart(self.length / GOLDEN_RATIO,
		                       (second_new_vertex, vertex_1, first_new_vertex))
		# Likewise create the two half kites
		half_kite_1 = P2HalfKite(self.length / GOLDEN_RATIO,
		                         (vertex_3, first_new_vertex, second_new_vertex))
		half_kite_2 = P2HalfKite(self.length / GOLDEN_RATIO,
		                         (vertex_3, first_new_vertex, vertex_2))
		# Return the three new triangles and any new vertices
		return [half_dart, half_kite_1, half_kite_2], new_vertices


class P2HalfDart(Tile):
	"""
	A triangle given by one half of the Dart tile from the P2 pattern

	The edge described by the first two elements of the vertices tuple
	is the diagonal of the whole dart tile, so these vertices should not
	be connected by an edge. That described by the second and third should
	form the longer external side and that described by the third and first
	should form the shorter side of the overall dart. Their lengths
	relative to this object's length attribute should be 1, phi and 1
	respectively.

	Deflation yields one P2HalfKite and one P2HalfDart.
	"""
	
	def deflate_once(self):
		vertex_1, vertex_2, vertex_3 = self.vertices
		# Create a vertex along the second edge such that it is phi times
		# further from the second vertex than the third. The neighbouring
		# tile may have already been deflated, so we need to check if
		# such a vertex has already been created and use that if so.
		# For consistency with the case where it does already exist, a
		# new vertex must therefore contain references to the relevant
		# existing vertices on the shared edge.
		new_vertex_pos = ((GOLDEN_RATIO*vertex_3.position + 1*vertex_2.position)
		                  / (1+GOLDEN_RATIO))
		new_vertex = TilingVertex(new_vertex_pos, [vertex_2])
		vertex_already_exists = False
		for v in vertex_2.adjacent_vertices:
			if v.distance_from(new_vertex) < self.length / 100:
				# All side lengths in the tiling are of order self.length,
				# so something within self.length/100 must be a duplicate
				# of the vertex we just created
				new_vertex = v
				vertex_already_exists = True
				new_vertices = []
		if not vertex_already_exists:
			vertex_2.adjacent_vertices.append(new_vertex)
			new_vertices = [new_vertex]
			# Disconnect the 2nd and 3rd vertices because there is now a
			# vertex on the edge between them
			vertex_2.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_2)
		# Connect the new vertex to vertex_1
		new_vertex.adjacent_vertices.append(vertex_1)
		vertex_1.adjacent_vertices.append(new_vertex)
		# Create the new half dart and half kite
		half_dart = P2HalfDart(self.length / GOLDEN_RATIO,
		                       (new_vertex, vertex_3, vertex_1))
		half_kite = P2HalfKite(self.length / GOLDEN_RATIO,
		                       (vertex_2, vertex_1, new_vertex))
		# Return the two new triangles and the new vertex if it is indeed new
		return [half_dart, half_kite], new_vertices


def p3_initial_tiling(target_tile_length, min_radius):
	"""
	Generates an undeflated initial tiling for the p3 pattern.

	Consists of a single fat rhombus centred on (0,0), with the longer
	diagonal along the x-axis. The rhombus is large enough that it
	completely covers a circle of radius min_radius centred on the origin
	and sized so that it can be deflated to give a tiling with a tile_length
	of target_tile_length.

	Returns two values: the initial tiling as a Tiling object and an
	integer containing the number of deflations required to reach the
	specified tile length. The Tiling object has a dummy defined_region
	value which always returns True.
	"""
	# This value will be useful
	sin_36_degrees = np.sqrt(1 - GOLDEN_RATIO**2 / 4)
	# The minimum distance from the origin to the edge of the tiling
	# is the perpendicular distance to one of the edges. This must be at
	# least equal to min_radius (add a 1% margin to account for any slight
	# innacuracies in the floating point arithmetic - the precise percentage
	# is unimportant).
	min_tile_length = 1.01 * min_radius / ((GOLDEN_RATIO / 2) * sin_36_degrees)
	# Find the minimum power of phi that gives a value greater than this
	# when multiplied by target_tile_length, and hence determine the correct
	# value for tile_length for this initial tiling
	num_deflations = math.ceil(np.log(min_tile_length / target_tile_length)
								/ np.log(GOLDEN_RATIO))
	tile_length = target_tile_length * GOLDEN_RATIO**num_deflations
	# Create the initial vertices accordingly
	top_vertex = TilingVertex(tile_length * np.array([0, sin_36_degrees]))
	bttm_vertex = TilingVertex(tile_length * np.array([0, -sin_36_degrees]))
	left_vertex = TilingVertex(
		(GOLDEN_RATIO/2) * tile_length * np.array([1, 0]),
		[top_vertex, bttm_vertex]
	)
	right_vertex = TilingVertex(
		(GOLDEN_RATIO/2) * tile_length * np.array([-1, 0]),
		[top_vertex, bttm_vertex]
	)
	top_vertex.adjacent_vertices = [left_vertex, right_vertex]
	bttm_vertex.adjacent_vertices = [left_vertex, right_vertex]
	# Create the 2 half-fat-rhombi
	upper_half_rhomb = P3HalfFatRhomb(tile_length,
	                                  (right_vertex, left_vertex, top_vertex))
	lower_half_rhomb = P3HalfFatRhomb(tile_length,
	                                  (right_vertex, left_vertex, bttm_vertex))
	# Create the initial Tiling object and return
	tiling = Tiling(
		[upper_half_rhomb, lower_half_rhomb],
		[top_vertex, bttm_vertex, left_vertex, right_vertex],
		(lambda x: True))
	return tiling, num_deflations


class P3HalfThinRhomb(Tile):
	"""
	A triangle given by one half of the thin rhombus tile from the P3 pattern

	The edge described by the first two elements of the vertices tuple
	is the tile's shorter diagonal, so these vertices should not be connected
    by an edge. That described by the second and third is the edge of the type
	which matches the third and second vertices (order matters) of a
	P3HalfFatRhomb object under the matching rules, and likewise that
	described by the third and first corresponds to the the third and first
	of a P3HalfFatRhomb. Their lengths relative to this object's length
	attribute should be (phi-1), 1 and 1 respectively.

	Deflation yields one P3HalfThinRhomb and one P3HalfFatRhomb.
	"""

	def deflate_once(self):
		vertex_1, vertex_2, vertex_3 = self.vertices
		# Create a vertex along the second edge such that it is phi times
		# further from the third vertex than the second. The neighbouring
		# tile may have already been deflated, so we need to check if
		# such a vertex has already been created and use that if so.
		# The new vertex should be connected vertex_1 and vertex_3, but not
		# vertex_2 (that edge is the short diagonal of a new thin rhombus).
		new_vertex_pos = ((GOLDEN_RATIO*vertex_2.position + 1*vertex_3.position)
		                  / (1+GOLDEN_RATIO))
		new_vertex = TilingVertex(new_vertex_pos, [vertex_1, vertex_3])
		vertex_already_exists = False
		for v in vertex_3.adjacent_vertices:
			if v.distance_from(new_vertex) < self.length / 100:
				# All side lengths in the tiling are of order self.length,
				# so something within self.length/100 must be a duplicate
				# of the vertex we just created
				new_vertex = v
				vertex_already_exists = True
				new_vertices = []
				assert vertex_3 in v.adjacent_vertices
				# We also need to add the connection to vertex_1 which
				# won't have been added during the deflation of the
				# neighbouring tile
				v.adjacent_vertices.append(vertex_1)
				vertex_1.adjacent_vertices.append(v)
		if not vertex_already_exists:
			# The vertex doesn't already exist, so we'll use the new one
			# we just made
			new_vertices = [new_vertex]
			# Finish up the new vertex's connections
			vertex_1.adjacent_vertices.append(new_vertex)
			vertex_3.adjacent_vertices.append(new_vertex)
			# Disconnect the 2nd and 3rd vertices because there is now a
			# vertex on the edge between them
			vertex_2.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_2)
		# Connect up the first two vertices (because this is now an edge,
		# not a tile's diagonal) if the neighbouring tile hasn't already
		# done so
		if vertex_2 not in vertex_1.adjacent_vertices:
			vertex_1.adjacent_vertices.append(vertex_2)
			vertex_2.adjacent_vertices.append(vertex_1)
		else:
			assert vertex_1 in vertex_2.adjacent_vertices
		# Disconnect the first and third vertices (because this is now a
		# fat rhombus' diagonal) if the neighbouring tile hasn't already
		# done so
		if vertex_3 in vertex_1.adjacent_vertices:
			vertex_1.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_1)
		else:
			assert vertex_1 not in vertex_3.adjacent_vertices
		# Create the new half rhombi
		half_thin_rhomb = P3HalfThinRhomb(self.length / GOLDEN_RATIO,
		                                  (vertex_2, new_vertex, vertex_1))
		half_fat_rhomb = P3HalfFatRhomb(self.length / GOLDEN_RATIO,
		                                (vertex_3, vertex_1, new_vertex))
		# Return the two new triangles and the new vertex if it is indeed new
		return [half_thin_rhomb, half_fat_rhomb], new_vertices


class P3HalfFatRhomb(Tile):
	"""
	A triangle given by one half of the fat rhombus tile from the P3 pattern

	The edge described by the first two elements of the vertices tuple
	is the tile's longer diagonal, so these vertices should not be connected
    by an edge. That described by the second and third is the edge of the type
	which matches the third and second vertices (order matters) of a
	P3HalfThinRhomb object under the matching rules, and likewise that
	described by the third and first corresponds to the the third and first
	of a P3HalfThinRhomb. Their lengths relative to this object's length
	attribute should be phi, 1 and 1 respectively.

	Deflation yields one P3HalfThinRhomb and two P3HalfFatRhombs.
	"""

	def deflate_once(self):
		vertex_1, vertex_2, vertex_3 = self.vertices
		# Create a vertex along the first edge such that it is phi times
		# further from the second vertex than the first. The neighbouring
		# tile may have already been deflated, so we need to check if
		# such a vertex has already been created and use that if so.
		# The new vertex should be connected to vertex_1 and vertex_3, but not
		# vertex_2 (that edge is the long diagonal of a new fat rhombus).
		new_vertex_pos = ((GOLDEN_RATIO*vertex_1.position + 1*vertex_2.position)
		                  / (1+GOLDEN_RATIO))
		new_vertex_1 = TilingVertex(new_vertex_pos, [vertex_1, vertex_3])
		vertex_already_exists = False
		for v in vertex_1.adjacent_vertices:
			if v.distance_from(new_vertex_1) < self.length / 100:
				# All side lengths in the tiling are of order self.length,
				# so something within self.length/100 must be a duplicate
				# of the vertex we just created
				new_vertex_1 = v
				vertex_already_exists = True
				new_vertices = []
				assert vertex_1 in v.adjacent_vertices
				# We also need to add the connection to vertex_3 which
				# won't have been added during the deflation of the
				# neighbouring tile
				v.adjacent_vertices.append(vertex_3)
				vertex_3.adjacent_vertices.append(v)
		if not vertex_already_exists:
			# The vertex doesn't already exist, so we'll use the new one
			# we just made
			new_vertices = [new_vertex_1]
			# Finish up the new vertex's connections (for now --- we'll
			# need to add a connection to the second new vertex later)
			vertex_1.adjacent_vertices.append(new_vertex_1)
			vertex_3.adjacent_vertices.append(new_vertex_1)
		# Repeat for a vertex along the second edge, closer to the third
		# vertex than the second.
		# This new vertex should be connected to vertex_2, but not vertex_3
		# (that edge is the short diagonal of a new thin rhombus).
		new_vertex_pos = ((GOLDEN_RATIO*vertex_3.position + 1*vertex_2.position)
		                  / (1+GOLDEN_RATIO))
		new_vertex_2 = TilingVertex(new_vertex_pos, [vertex_2])
		vertex_already_exists = False
		for v in vertex_2.adjacent_vertices:
			if v.distance_from(new_vertex_2) < self.length / 100:
				# All side lengths in the tiling are of order self.length,
				# so something within self.length/100 must be a duplicate
				# of the vertex we just created
				new_vertex_2 = v
				vertex_already_exists = True
				assert vertex_2 in v.adjacent_vertices
		if not vertex_already_exists:
			# The vertex doesn't already exist, so we'll use the new one
			# we just made
			new_vertices.append(new_vertex_2)
			# Finish up the new vertex's connection (for now --- we'll
			# need to add a connection to the second new vertex later)
			vertex_2.adjacent_vertices.append(new_vertex_2)
			# Disconnect the 2nd and 3rd vertices because there is now a
			# vertex on the edge between them
			vertex_2.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_2)
		# Connect the two new vertices together
		new_vertex_1.adjacent_vertices.append(new_vertex_2)
		new_vertex_2.adjacent_vertices.append(new_vertex_1)
		# Disconnect the first and third vertices (because this is now a
		# fat rhombus' diagonal) if the neighbouring tile hasn't already
		# done so
		if vertex_3 in vertex_1.adjacent_vertices:
			vertex_1.adjacent_vertices.remove(vertex_3)
			vertex_3.adjacent_vertices.remove(vertex_1)
		else:
			assert vertex_1 not in vertex_3.adjacent_vertices
		# Create the new half rhombi
		half_thin_rhmb = P3HalfThinRhomb(self.length / GOLDEN_RATIO,
		                                  (vertex_3, new_vertex_2, new_vertex_1))
		half_fat_rhmb_1 = P3HalfFatRhomb(self.length / GOLDEN_RATIO,
		                                  (vertex_2, new_vertex_1, new_vertex_2))
		half_fat_rhmb_2 = P3HalfFatRhomb(self.length / GOLDEN_RATIO,
		                                  (vertex_3, vertex_1, new_vertex_1))
		# Return the three new triangles and the new vertices (if any)
		return [half_thin_rhmb, half_fat_rhmb_1, half_fat_rhmb_2], new_vertices


def norm(vector):
	"""
	Returns the magnitude of a vector (passed as a 1D numpy array)
	"""
	return np.sqrt(np.square(vector).sum())
