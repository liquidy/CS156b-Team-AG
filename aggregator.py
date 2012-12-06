from numpy import matrix
from numpy import linalg
import string
import time

predictors = [
              'basicsvd_20_knn', \
              'svd_200_knn', \
              'svd_1000_knn', \
              'svdp_800_knn', \
              'timesvd_100_knn', \
              'basicsvd_50', \
              # 'svd_5', \
              # 'svd_20', \
              'svd_50', \
              'svd_100', \
              # 'svd_200', \
              'svd_1000', \
              # 'svd_individual_50', \
              'svdp_50', \
              'svdp_200', \
              'svdp_800', \
              'timesvd_20', \
              'timesvd_100', \
              # 'timesvd_170', \
              'timesvd_350', \
]
# predictors = [
#               # 'knn_ratings_50', \
#               'basicsvd_20_knn', \
#               'svd_200_knn', \
#               'svd_1000_knn', \
#               'svdp_800_knn', \
#               'timesvd_100_knn', \
#               'basicsvd_50', \
#               'svd_5', \
#               'svd_20', \
#               'svd_50', \
#               'svd_100', \
#               'svd_200', \
#               'svd_1000', \
#               # 'svd_individual_20', \
#               'svd_individual_50', \
#               # 'svdp_20', \
#               'svdp_50', \
#               'svdp_200', \
#               'svdp_800', \
#               # 'svdp_flip_20', \
#               # 'svdp_flip_200', \
#               'timesvd_20', \
#               'timesvd_100', \
#               'timesvd_170', \
#               'timesvd_350', \
# ]
q_mean = 3.6749
G_T = []
R_T = []

# Read in probe predictor ratings
print 'Reading in %d predictor probe ratings:' % len(predictors)
for p in predictors:
    print '   ', p
    g = []
    rating_sum = 0
    f_predictor = open(p + '_probe')
    for line in f_predictor:
        r = float(line)
        rating_sum += r
        g.append(r)
    p_mean = float(rating_sum) / len(g)
    for i in range(len(g)):
        g[i] -= p_mean
    G_T.append(g)

# Read in actual probe ratings
print 'Reading in probe.'
rating_sum = 0
f_probe = open('probe.dta')
for line in f_probe:
    r = float(line)
    rating_sum += r
    R_T.append(r)
probe_mean = float(rating_sum) / len(R_T)
probe = R_T[:]
for i in range(len(R_T)):
    R_T[i] -= probe_mean

# Calculate weights
print 'Calculating weights.'
R = matrix(R_T).T
G_T = matrix(G_T)
G = G_T.T
weights = ((G_T*G).I)*(G_T*R)
print '\n', weights, '\n'

# Read in predictor ratings
print 'Reading in predictor qual ratings.'
predictors_qual = []
means = []
for i in range(len(predictors)):
    predictors_qual.append([])
for i in range(len(predictors)):
    rating_sum = 0
    predicted_ratings = open(predictors[i])
    for line in predicted_ratings:
        r = float(line)
        rating_sum += r
        predictors_qual[i].append(r)
    means.append(float(rating_sum) / len(predictors_qual[i]))
    for j in range(len(predictors_qual[i])):
        predictors_qual[i][j] -= means[i]

# Generate blended output
print 'Generating output now.'
f_output = open('blend', 'w')
for i in range(2749898):
    rating_sum = q_mean
    for j in range(len(predictors_qual)):
        r = predictors_qual[j][i]
        w = weights[j]
        rating_sum += w * r
    if rating_sum > 5:
        rating_sum = 5
    elif rating_sum < 1:
        rating_sum = 1
    f_output.write('%.4f\n' % rating_sum)
    if i % 274989 == 0:
        print string.join(time.ctime().split()[3:4], '') \
            + ': ', i, '/ 2749898'
f_output.close()





