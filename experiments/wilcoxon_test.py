import scipy.stats as stats


mymodel = [0.57, 0.66, 0.655, 0.845, 0.635, 0.905, 0.775, 0.8, 0.545, 0.575, 0.59, 0.745, 0.615, 0.69, 0.89, 0.62, 0.6, 0.53]
group2 = [0.56, 0.64, 0.575, 0.73, 0.575, 0.825, 0.745, 0.77, 0.465, 0.48, 0.59, 0.71, 0.59, 0.5, 0.83, 0.605, 0.5, 0.48]

print(stats.wilcoxon(mymodel, group2))