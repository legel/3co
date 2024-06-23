from scipy.stats import beta
import numpy as np

def n_samples_from_distribution(n_samples=10, sample_lower_bound=-10.0, sample_upper_bound=10.0, mean=0.0, variance=1.0, percentile_lower_bound=0.00, percentile_upper_bound=1.00, min_sample_distance_as_percent_of_mean=0.025):
	# normalize mean and variance as if distribution was between 0.0 - 1.0
	sample_range = sample_upper_bound - sample_lower_bound
	normalized_mean = (mean - sample_lower_bound) / sample_range
	normalized_variance = variance / sample_range

	# https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
	a = ((1 - normalized_mean) / normalized_variance**2 - 1/normalized_mean) * normalized_mean**2
	b = a*(1/normalized_mean - 1)

	#print("a = {}".format(a))
	#print("b = {}\n".format(b))

	percentiles_to_sample = np.linspace(percentile_lower_bound, percentile_upper_bound, n_samples)
	#sampled_values = norm.ppf(percentiles_to_sample, loc=mean, scale=variance)
	sampled_values = beta.ppf(percentiles_to_sample, a=a, b=b)
	sampled_densities = beta.pdf(percentiles_to_sample, a=a, b=b)

	converted_sample_values = [sample_value * sample_range + sample_lower_bound for sample_value in sampled_values]

	# print("Sample range: {}".format(sample_range))
	# print("Normalized mean: {}".format(normalized_mean))
	# print("Normalized variance: {}".format(normalized_variance))

	#for percentile, sampled_value, sample_density in zip(percentiles_to_sample, converted_sample_values, sampled_densities):
	#	print("{:.1f}% - {:.1f}".format(percentile * 100, sampled_value))

	min_sample_distance_in_original_space = min_sample_distance_as_percent_of_mean * mean
	accepted_samples = [converted_sample_values[0]]
	for sample in converted_sample_values[1:]:
		if sample - accepted_samples[-1] > min_sample_distance_in_original_space:
			accepted_samples.append(sample)

	#print("\n...with the following accepted:")

	#for percentile, sampled_value in zip(percentiles_to_sample, accepted_samples):
	#	print("{:.1f}% - {:.1f}".format(percentile * 100, sampled_value))

	return accepted_samples

# print("Example optical focus distance values: ")
# converted_sample_values = n_samples_from_distribution(n_samples=11, sample_lower_bound=200.0, sample_upper_bound=600.0, mean=450.0, variance=25.0, percentile_lower_bound=0.00, percentile_upper_bound=1.0, min_sample_distance_as_percent_of_mean=0.05)

# print("\nExample exposure times: ")
# converted_sample_values = n_samples_from_distribution(n_samples=20, sample_lower_bound=10.0, sample_upper_bound=350.0, mean=100.0, variance=100.0, percentile_lower_bound=0.00, percentile_upper_bound=1.0)
