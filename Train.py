import math
import pickle
import sys

# Variables and constants
num_users = 458293
num_movies = 17770

num_features = 1
lrate = 1e-3
num_epochs = 20
#min_err_decrease = 5e-6

userFeature = [[0 for x in xrange(num_users)] for x in xrange(num_features)] 
movieFeature = [[0 for x in xrange(num_movies)] for x in xrange(num_features)] 



def predict_rating(movie, user, featureBeingTrained):
    # Calculate rating_sum
    rating_sum = userFeature[featureBeingTrained][user] * \
            movieFeature[featureBeingTrained][movie]
    return rating_sum


# Train for 'num_features' features
for featureBeingTrained in range(num_features):
    # Initialize the feature vector
    userFeature[featureBeingTrained] = [2.0 / math.sqrt(num_features)] * num_users
    movieFeature[featureBeingTrained] = [2.0 / math.sqrt(num_features)] * num_movies
    userValue = userFeature[featureBeingTrained]
    movieValue = movieFeature[featureBeingTrained]

    # Traverse input and train for this particular feature
    input_data = open('mu/all.dta', 'r')
    i = 0
    for line in input_data:
        line = line.strip()
        user, movie, date, rating = line.split(' ')
        user, movie, date, rating = int(user) - 1, int(movie) - 1, int(date), int(rating)

        # Train on this data point a few times.
        # TODO: switch this to: continue until error is less than ...
        # err_diff = sys.maxint
        # old_err = sys.maxint
        for j in range(num_epochs):
        # while err_diff > min_err_decrease:
            err = lrate * (rating - predict_rating(movie, user, featureBeingTrained))
            uv = userValue[user]
            userValue[user] += err * movieValue[movie]
            movieValue[movie] += err * uv

            # err_diff = abs(old_err - err)
            # old_err = err
        if i % 10000 == 0:
            print 'training feature vector', featureBeingTrained, 'for point', i
        i += 1
    input_data.close()
    print 'feature', featureBeingTrained, 'done'

# Done! Write the userFeature and movieFeature arrays out via pickling
userFeature_output = open('userFeature', 'wb')
pickle.dump(userFeature, userFeature_output)
userFeature_output.close()
movieFeature_output = open('movieFeature', 'wb')
pickle.dump(movieFeature, movieFeature_output)
movieFeature_output.close()


