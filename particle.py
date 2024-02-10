"""
Defines the type Particle to represent a particle performing discrete
Monte-Carlo hops on a lattice (as defined in the lattice module), and the
type Trajectory to represent its position over time and calculate the
resulting HeSE data.
"""


import numpy as np
import scipy.fft, scipy.optimize
import matplotlib.pyplot as plt


MAX_ISF_CURVE_FIT_ITERATIONS = 1000


class Particle:
	"""
	Object to represent a particle confined to a lattice, and the random
	hopping thereof.

	Attributes:
	current_location    The instance of lattice.LatticePoint representing
	                    the particle's current location on the lattice
	parent_lattice      The overall lattice.Lattice on which the particle
	                    is hopping
	hopping_algorithm   A callable which controls the probabilities with
	                    which the particle hops. Takes the current
	                    LatticePoint and a random float uniformly distributed
	                    in [0,1) (for convenience and performance; it can
	                    still do its own rng) as arguments (in that order),
	                    and returns the new location of the particle as a
	                    LatticePoint (returns its first argument if
	                    no hop occurs).
	time_step           The time interval corresponding to each simulation
	                    step
	"""

	def __init__(self, lattice, start_loc, hop_alg=None, time_step=1, p_hop=.5):
		"""
		Arguments:
		lattice             The particle's parent lattice
		start_loc           The lattice.LatticePoint corresponding to
		                    the particle's initial position
		hop_alg             The callable for the particle's hopping_algorithm
		                    attribute. If unspecified, the particle will
		                    hop with the same overall probability from
		                    every lattice point, picking each neighbour
		                    with equal probability (giving hopping rates
		                    inversely proportional to the site's number
		                    of neighbours).
		p_hop               The overall probability that the particle will
		                    hop in a given step when using the default
		                    algorithm (ignored if hop_alg is specified)
		time_step           The time step for the new particle
		"""
		self.time_step = time_step
		self.parent_lattice = lattice
		self.current_location = start_loc
		if hop_alg is None:
			hop_alg = (lambda p, rnd:
				p if rnd >= p_hop
				else p.nrst_neighbours[int(len(p.nrst_neighbours)*rnd/p_hop)]
			)
		self.hopping_algorithm = hop_alg

	def copy(self):
		"""
		Returns a new instance of Particle with the same attributes as this one.
		"""
		return Particle(self.parent_lattice,
		                self.current_location,
		                self.hopping_algorithm,
		                self.time_step)

	def simulate_hopping(self, num_steps):
		"""
		Simulates the random hopping process for num_steps steps and
		returns the particle's trajectory as an instance of Trajectory
		"""
		# Create a new numpy.random.generator instance
		rng = np.random.default_rng()
		# Create an array of random numbers in [0,1), which will be passed
		# to the hopping algorithm
		random_nums = rng.random(num_steps)
		# Create an empty array for the trajectory and enter the current
		# position as the first value
		traj = np.zeros((num_steps+1,2))
		traj[0] = self.current_location.position
		# Loop through the steps, adding each position to the trajectory
		for step_num in range(0, num_steps):
			# Perform hop according to specified algorithm
			self.current_location = self.hopping_algorithm(
				self.current_location, random_nums[step_num]
			)
			# Check we haven't run off the end of the lattice
			if not self.parent_lattice.is_in_defined_region(self.current_location.position):
				raise RuntimeError(
					"Particle has left the region where the lattice is properly defined"
				)
			# Append new position to the trajectory
			traj[step_num+1] = self.current_location.position
		# Return the trajectory as an instance of Trajectory
		return Trajectory(self.time_step, traj[:,0], traj[:,1])

	def simulate_ensemble(self, num_steps, num_simulations, print_progress=False):
		"""
		Runs multiple simulations starting from the particle's current state.

		Returns an instance of TrajectoryEnsemble containing num_simulations
		simulations. The particle is used as a template, so is not itself
		modified at all.
		"""
		trajectories = []
		for i in range(num_simulations):
			if print_progress: print(f"\r{100*i/num_simulations:.2f}%", end="")
			trajectories.append(self.copy().simulate_hopping(num_steps))
		if print_progress: print("\r100.00%")
		return TrajectoryEnsemble(trajectories)


def equal_hopping_rates_algrthm(rate, time_step=1):
	"""
	A convenience function generate a hopping_algorithm attribute for a
	Particle object corresponding to every (possible) jump having the same
	hopping rate.

	Returns a callable which can be passed directly into the instantiation
	method of Particle.
	"""
	prob_per_step = rate * time_step
	max_neighbours = int(1 / prob_per_step)
	def func(point, rand_num):
		num_nghbrs = len(point.nrst_neighbours)
		if num_nghbrs > max_neighbours:
			raise ValueError(
				  f"A lattice point had {num_nghbrs} neighbours, which "
				+ f"makes a hopping rate of {rate} for all of them "
				+ f"impossible with a time step of {time_step}"
			)
		if rand_num >= prob_per_step * num_nghbrs:
			return point
		else:
			return point.nrst_neighbours[int(rand_num / prob_per_step)]
	return func


class Trajectory:
	"""
	Object to represent the path of a particle over the course of a
	number of discrete time steps.

	Attributes:
	t                   Array of time values
	x                   Array of corresponding x coordinates
	y                   Array of corresponding y coordinates
	"""

	def __init__(self, time_step, x, y):
		self.t, self.x, self.y = time_step * np.arange(len(x)), x, y

	def scattered_amplitude(self, delta_k, step_num=None):
		"""
		Returns an array of (complex) scattering amplitudes corresponding
		to the momentum transfer vectors specified.
		
		delta_k should be a 2D array(-like) with length 2 in the 0 dimension, such
		that delta_k[0] contains the x components of the vectors and
		delta_k[1] contains the y components.

		When step_num is None, the returned array has two dimensions,
		with each row (ie an index along the 0	axis) corresponding to one
		of the times specified in the trajectory's t attribute and each
		column (ie an index along the 1 axis) corresponding to one of the
		momentum transfer vectors specified in the delta_k argument. When 
		step_num is not None, the returned array has only one dimension,
		corresponding to A(Delta K) evaluated at the time t[step_num].
		"""
		# Calculate the values of delta_k[dotproduct]r(t)
		if step_num is None:
			dot_prod = np.outer(self.x, delta_k[0]) + np.outer(self.y, delta_k[1])
		else:
			dot_prod = self.x[step_num] * delta_k[0] + self.y[step_num] * delta_k[1]
		# Return the complex exponential thereof
		return np.exp(-1j*dot_prod)

	def pair_correlation_function(self, t, pos_tol=1e-5):
		"""
		Returns the pair correl. function for the trajectory at time value t

		The pair correlation function consists of a set of delta functions
		with different coefficients and positions in (real) 2D space, so
		this method returns a dictionary with cartesian position vectors
		as tuples for keys and the corresponding coefficients as values.

		t must correspond to a multiple of the temporal sampling period
		(ie the time_step instantiation argument).

		To account for floating point errors, delta functions with
		positions a distance less than about pos_tol apart will be merged.
		"""
		peak_dict = {}
		time_step = self.t[1] - self.t[0]
		step_diff = t/time_step
		if not abs(int(step_diff+1e-9) - step_diff) < 2e-9:
			print(time_step, t, step_diff)
			raise ValueError(
				"t must be a multiple of the temporal sampling period"
			)
		step_diff = int(step_diff+1e-9)
		# Count the number of times each displacement occurs at the given
		# time separation
		x_diplacements = self.x[step_diff:] - self.x[:len(self.x)-step_diff]
		y_diplacements = self.y[step_diff:] - self.y[:len(self.y)-step_diff]
		for displ_tuple in zip(x_diplacements, y_diplacements):
			try:
				peak_dict[displ_tuple] += 1
			except KeyError:
				peak_dict[displ_tuple] = 1
		# Merge positions within tolerance of each other
		return_dict = {}
		for dx, dy in peak_dict:
			key = next(
				(
					v for v in peak_dict
					if max(abs(v[0]-dx), abs(v[1]-dy)) <= pos_tol
				),
				(dx, dy)
			)
			try:
				return_dict[key] += peak_dict[dx,dy]
			except KeyError:
				return_dict[key] = peak_dict[dx,dy]
		# Divide through by the total number to get probabilities
		for key in return_dict:
			return_dict[key] = return_dict[key] / (len(self.t) - step_diff)
		return return_dict

	def intermediate_scattering_function(self, delta_k):
		"""
		Returns the ISF for the trajectory at a specific delta_k value.

		The intermediate scattering function is the time autocorrelation
		of the scattered amplitude. The amplitude is (somewhat) arbitrary,
		so this method normalises it by dividing by the number of simulation
		steps (such that I(delta_k, 0) = 1).

		delta_k should be a 2-element array-like, containing the x and y
		components of the momentum transfer vector.

		Returns 2 arrays; the first contains values of t and the second
		is the corresponding values of I(delta_k, t) .
		"""
		# The output times are the same as the trajectory's but shifted
		# to be centred on zero
		t = self.t - self.t[-1]/2
		# Calcalate autocorrelation as inverse FT of modulus squared of the FT
		amplitude = self.scattered_amplitude([[delta_k[0]], [delta_k[1]]])[:,0]
		mod_ft_sqrd = np.abs(scipy.fft.fft(amplitude))**2
		return t, scipy.fft.fftshift(scipy.fft.ifft(mod_ft_sqrd)) / len(t)

	def ISF_snapshots(self, delta_k_vectors, t_vals, print_progress=False):
		"""
		Returns the ISF for the trajectory at specified t and delta_k values.

		In contrast to intermediate_scattering_function(), this method is
		optimised for a small number of time values but many delta_k values.

		delta_k should be an array-like of 2-element array-likes, each
		containing the x and y components of a delta_k vector, while
		t_vals should be a 1D array-like containing the time values. The
		time values should be multiples of the simulation step length.

		Returns a 2D array such that the element array[j][i] contains the
		value of I(delta_k_vectors[i], t_vals[j]).
		"""
		k_x, k_y = np.array(delta_k_vectors).T
		pcfs = []
		for i, t in enumerate(t_vals):
			pcfs.append(self.pair_correlation_function(t))
			if print_progress:
				print(f"\r{100*i/len(t_vals):.2f}%", end="")
		if print_progress: print("\r100.00%")
		return np.array([
			sum(
				pcf[v] * np.exp(- 1j * k_x * v[0] - 1j * k_y * v[1])
				for v in pcf
			)
			for pcf in pcfs
		])

	def ISF_and_fit(self, delta_k, fit_func, fit_region=None, initial_param_guess=None):
		"""
		Returns the intermediate scattering function and the parameters
		required to fit the function fit_func thereto.

		The first two return values are the time and intermediate scattering
		function values for this trajectory and the specified delta_k
		value, as documented in the intermediate_scattering_function()
		method.

		The third return value is an array of the parameters for which
		fit_func gives the optimal fit for t values in the interval
		[fit_region[0], fit_region[1]] (or [0, fit_region] if fit_region is
		not subscriptable). fit_func must be a model function as documented
		for scipy.optimize.curve_fit(). If fit_region is not specified,
		all available t values will be used.

		initial_param_guess is used as the p0 argument of the scipy.curve_fit()
		function (so should be an array like containing initial guesses
		for all parameters).
		"""
		t, I = self.intermediate_scattering_function(delta_k)
		# Create copies of t and I covering only the required t range
		t_to_fit, I_to_fit = t[int(len(t)/2):], I[int(len(I)/2):]
		if fit_region is not None:
			try:
				start_t, stop_t = fit_region[0], fit_region[1]
			except TypeError:
				start_t, stop_t = 0, fit_region
			start_index = next(
				(i for (i, t_val) in enumerate(t_to_fit) if t_val >= start_t),
				None
			)
			stop_index = next(
				(i for (i, t_val) in enumerate(t_to_fit) if t_val > stop_t),
				None
			)
			if start_index is None or start_index > stop_index:
				raise ValueError("No t values lie within the specified fit_region")
			if stop_index is not None:
				t_to_fit = t_to_fit[start_index : stop_index]
				I_to_fit = I_to_fit[start_index : stop_index]
		params, _ = scipy.optimize.curve_fit(fit_func,
		                                     t_to_fit,
		                                     np.real(I_to_fit),
		                                     initial_param_guess,
		                                     maxfev=MAX_ISF_CURVE_FIT_ITERATIONS)
		return t, I, params

	def fit_parameters(
		self, delta_k_vectors, fit_func, fit_region=None,
		initial_param_guess=None, print_progress=False
	):
		"""
		Returns an array of the optimal parameters from fitting fit_func
		to the ISF for the specified delta_k_vectors.

		The returned array is 2 dimensional, with array[i][j] corresponding
		to the optimal value of the ith parameter for the ISF produced
		by the jth delta_k vector.

		Arguments:
			delta_k_vectors A 2D array(-like) of 2-element array(-like)s.
			                ie an array with length 2 in the 1 dimension,
			                such that delta_k_vectors[i][0] contains the
			                x component of the ithdelta_k value.
			fit_func        A callable which takes an array of t values
			                and any number of parameters with the signature
			                fit_func(t, a, b,...) (where a, b,... are
			                parameters), and outputs an array of fitted
			                ISF values.
			fit_region      The fitting will only be performed in the
			                time range [fit_region[0], fit_region[1]]
			                (or [0, fit_region] if fit_region is not
			                subscriptable). Values outside of this will
			                be ignored. If fit_region is not specified,
			                all available t values will be used.
			print_progress  Whether to print progress to stdout as a
			                percentage.
		"""
		param_arrays = []
		for i, delta_k in enumerate(delta_k_vectors):
			if print_progress: print(f"\r{100*i/len(delta_k_vectors):.2f}%", end="")
			param_arrays.append(self.ISF_and_fit(
				delta_k, fit_func, fit_region, initial_param_guess)[2])
		if print_progress: print("\r100.00%")
		return np.array(param_arrays).transpose()

	def noisy_ISF(self, delta_k, peak_counts, t_vals):
		"""
		Returns the ISF sampled at the specified t values with simulated
		shot noise added.

		Returns an array of ISF values corresponding to the t values
		in the array-like t_vals. If these t values do not correspond to
		those of the simulated data, the ISF thereat is estimated by
		linear interpolation.

		The noise corresponds to the ISF being measured by counts at a
		detector, with the value I(k,0)=1 corresponding to peak_counts
		counts. In other words, the noisy ISF is Poisson distributed
		about the exact value of the simulated ISF, I, with a standard
		deviation of (I*peak_counts)^(-1/2).

		This method assumes the ISF is real.

		Arguments:
			delta_k         a 2-element array-like, containing the x and y
			                components of the momentum transfer vector.
			peak_counts     The number of detector counts corresponding
			                to a unit ISF value (ie its value at t=0).
			                A larger value will give less noisy data.
			t_vals          An array(-like) of t values at which to evaluate
			                the ISF (linearly interpolated from the
			                simulated data).
		"""
		sim_t, sim_ISF = self.intermediate_scattering_function(delta_k)
		interp_ISF = np.interp(t_vals, sim_t, np.real(sim_ISF))
		non_neg_ISF = np.max([interp_ISF, np.zeros(len(t_vals))], axis=0)
		rng = np.random.default_rng()
		ISF_as_counts = rng.poisson(peak_counts * non_neg_ISF)
		return ISF_as_counts / peak_counts
		
	def noisy_ISF_snapshots(
		self, delta_k_vectors, t_vals, peak_counts, print_progress=False
	):
		"""
		Returns the noisy ISF for at specified t and delta_k values.

		In contrast to noisy_ISF(), this method is optimised for a small
		number of time values but many delta_k values.

		delta_k should be an array-like of 2-element array-likes, each
		containing the x and y components of a delta_k vector, while
		t_vals should be a 1D array-like containing the time values. The
		time values should be multiples of the simulation step length.

		Returns a 2D array such that the element array[j][i] contains the
		value of I(delta_k_vectors[i], t_vals[j]).

		The noise corresponds to the ISF being measured by counts at a
		detector, with the value I(k,0)=1 corresponding to peak_counts
		counts. In other words, the noisy ISF is Poisson distributed
		about the exact value of the simulated ISF, I, with a standard
		deviation of (I*peak_counts)^(-1/2).

		This method assumes the ISF is real.
		"""
		sim_ISF = np.real(self.ISF_snapshots(delta_k_vectors, t_vals, print_progress))
		non_neg_ISF = np.max([sim_ISF, np.zeros(sim_ISF.shape)], axis=0)
		rng = np.random.default_rng()
		ISF_as_counts = rng.poisson(peak_counts * non_neg_ISF)
		return ISF_as_counts / peak_counts

	def noisy_ISF_and_fit(
		self, delta_k, peak_counts, t_vals,
		fit_func, fit_region=None, initial_param_guess=None
	):
		"""
		Returns the intermediate scattering function with simulated shot noise,
		and the parameters required to fit the function fit_func thereto.

		The first two return values are the time and intermediate scattering
		function values for this trajectory and the specified delta_k
		peak_counts and t_vals, as documented in the noisy_ISF() method.

		The third return value is an array of the parameters for which
		fit_func gives the optimal fit for t values in the interval
		[fit_region[0], fit_region[1]] (or [0, fit_region] if fit_region is
		not subscriptable). fit_func must be a model function as documented
		for scipy.optimize.curve_fit(). If fit_region is not specified,
		all available t values will be used.

		initial_param_guess is used as the p0 argument of the scipy.curve_fit()
		function (so should be an array like containing initial guesses
		for all parameters).
		"""
		isf = self.noisy_ISF(delta_k, peak_counts, t_vals)
		# Create copies of t and I covering only the required t range
		if fit_region is not None:
			try:
				start_t, stop_t = fit_region[0], fit_region[1]
			except TypeError:
				start_t, stop_t = 0, fit_region
			start_index = next(
				(i for (i, t_val) in enumerate(t_vals) if t_val >= start_t),
				None
			)
			stop_index = next(
				(i for (i, t_val) in enumerate(t_vals) if t_val > stop_t),
				len(t_vals)
			)
			if start_index is None or start_index > stop_index:
				raise ValueError("No t values lie within the specified fit_region")
			t_to_fit = t_vals[start_index : stop_index]
			I_to_fit = isf[start_index : stop_index]
		else:
			t_to_fit, I_to_fit = t_vals, isf
		# Perform the fitting and return
		params, _ = scipy.optimize.curve_fit(
			fit_func,
			t_to_fit,
			np.real(I_to_fit),
			initial_param_guess,
			maxfev=MAX_ISF_CURVE_FIT_ITERATIONS
		)
		return t_vals, isf, params

	def noisy_fit_parameters(self, delta_k_vectors, peak_counts, t_vals, fit_func,
	                         initial_param_guess=None, print_progress=False):
		"""
		Returns an array of the optimal parameters from fitting fit_func
		to the ISF, with simulated noise, for the specified delta_k_vectors.

		The returned array is 2 dimensional, with array[i][j] corresponding
		to the optimal value of the ith parameter for the ISF produced
		by the jth delta_k vector.

		Arguments:
			delta_k_vectors A 2D array(-like) of 2-element array(-like)s.
			                ie an array with length 2 in the 1 dimension,
			                such that delta_k_vectors[i][0] contains the
			                x component of the ith delta_k value.
			peak_counts     The number of detector counts corresponding
			                to a unit ISF value (ie its value at t=0).
			                A larger value will give less noisy data.
			t_vals          An array of t values at which to generate
			                the ISF and perform the fitting.
			fit_func        A callable which takes an array of t values
			                and any number of parameters with the signature
			                fit_func(t, a, b,...) (where a, b,... are
			                parameters), and outputs an array of fitted
			                ISF values.
			print_progress  Whether to print progress to stdout as a
			                percentage.
		"""
		param_arrays = []
		for i, delta_k in enumerate(delta_k_vectors):
			if print_progress:
				print(f"\r{100*i/len(delta_k_vectors):.2f}%", end="")
			param_arrays.append(self.noisy_ISF_and_fit(
				delta_k, peak_counts, t_vals, fit_func, None, initial_param_guess
			)[2])
		if print_progress: print("\r100.00%")
		return np.array(param_arrays).transpose()

	def plot_spatial_trajectory(self, plot_size=None):
		"""
		Plots the spatial trajectory of the particle as a line
		"""
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.set_xlabel(r"$x/\AA$")
		ax.set_ylabel(r"$y/\AA$")
		if plot_size is not None:
			ax.set_xlim([-plot_size/2,plot_size/2])
			ax.set_ylim([-plot_size/2,plot_size/2])
		ax.set_title("Trajectory in space")
		ax.plot(self.x, self.y)
		plt.show()


class TrajectoryEnsemble:
	"""
	Object to represent a collection of trajectories from multiple
	different simulations.

	Methods are duplicates of those of Trajectory, but use suitable
	averages over the ensemble.

	Attributes:
	trajectories        A list of the Trajectory objects which make up
	                    the ensemble.
	"""

	def __init__(self, trajectories):
		if not all(all(trjctry.t == trajectories[0].t) for trjctry in trajectories):
			raise ValueError(
				"All trajectories in a TrajectoryEnsemble must have "
				+ "the same time values"
			)
		if len(trajectories) < 1:
			raise ValueError("TrajectoryEnsemble cannot be empty")
		self.trajectories = trajectories

	def intermediate_scattering_function(self, delta_k):
		"""
		Returns the ISF averaged over the trajectories at a specific delta_k value.

		The intermediate scattering function is the time autocorrelation
		of the scattered amplitude.

		delta_k should be a 2-element array-like, containing the x and y
		components of the momentum transfer vector.

		Returns 2 arrays; the first contains values of t and the second
		is the corresponding values of I(delta_k, t) .
		"""
		# The output times are the same as the trajectories' but shifted
		# to be centred on zero - use the zeroth since they're all the same
		t = self.trajectories[0].t - self.trajectories[0].t[-1]/2
		# Average the ISFs for the different trajectories
		# Use a loop because the np.average function gives out of memory
		# issues for large ensembles
		I = np.zeros(len(t), dtype=np.complex128)
		for traj in self.trajectories:
			I += traj.intermediate_scattering_function(delta_k)[1] / len(self.trajectories)
		return t, I

	def ISF_snapshots(self, delta_k_vectors, t_vals, print_progress=False):
		"""
		Returns the ISF for the trajectory at specified t and delta_k values.

		In contrast to intermediate_scattering_function(), this method is
		optimised for a small number of time values but many delta_k values.

		delta_k should be an array-like of 2-element array-likes, each
		containing the x and y components of a delta_k vector, while
		t_vals should be a 1D array-like containing the time values. The
		time values should be multiples of the simulation step length.

		Returns a 2D array such that the element array[j][i] contains the
		value of I(delta_k_vectors[i], t_vals[j]).
		"""
		I = np.zeros((len(t_vals), len(delta_k_vectors)), dtype=np.complex128)
		for i, traj in enumerate(self.trajectories):
			if print_progress:
				print(f"\r{100*i/len(self.trajectories):.2f}%", end="")
			I += traj.ISF_snapshots(delta_k_vectors, t_vals) / len(self.trajectories)
		if print_progress: print("\r100.00%")
		return I

	def pair_correlation_function(self, t, pos_tol=1e-5):
		"""
		Returns the pair correl. function averaged over the trajectories
		at time value t.

		The pair correlation function consists of a set of delta functions
		with different coefficients and positions in (real) 2D space, so
		this method returns a dictionary with cartesian position vectors
		as tuples for keys and the corresponding coefficients as values.

		t must correspond to a multiple of the temporal sampling period
		(ie the time_step instantiation argument).

		To account for floating point errors, delta functions with
		positions a distance less than about pos_tol apart will be merged.
		"""
		time_step = self.trajectories[0].t[1] - self.trajectories[0].t[0]
		denom = sum(len(traj.t) - t/time_step for traj in self.trajectories)
		return_dict = {}
		for traj in self.trajectories:
			pcf_dict = traj.pair_correlation_function(t, pos_tol)
			for dx,dy in pcf_dict:
				time_step = traj.t[1] - traj.t[0]
				val = pcf_dict[dx,dy] * (len(traj.t) - t/time_step) / denom
				key = next(
					(
						v for v in return_dict
						if max(abs(v[0]-dx), abs(v[1]-dy)) <= pos_tol
					),
					(dx, dy)
				)
				if key in return_dict:
					return_dict[key] += val
				else:
					return_dict[key] = val
		return return_dict

	ISF_and_fit = Trajectory.ISF_and_fit
	fit_parameters = Trajectory.fit_parameters
	noisy_ISF = Trajectory.noisy_ISF
	noisy_ISF_snapshots = Trajectory.noisy_ISF_snapshots
	noisy_ISF_and_fit = Trajectory.noisy_ISF_and_fit
	noisy_fit_parameters = Trajectory.noisy_fit_parameters
