import pickle

# Variables and constants
num_features = 40
num_users = 458293
num_movies = 17770
num_iter = 100
lrate = 0.001

userFeature = [[0 for x in xrange(num_users)] for x in xrange(num_features)] 
movieFeature = [[0 for x in xrange(num_movies)] for x in xrange(num_features)] 


def predict_rating(movie, user, up_to_feature):
	rating_sum = 0
	for i in range(up_to_feature):
		rating_sum += userFeature[i][user] * movieFeature[i][movie]
	return rating_sum


# Train for 'num_features' features
for featureBeingTrained in range(num_features):
	# Initialize the feature vector
	userFeature[featureBeingTrained] = [.1] * num_users
	movieFeature[featureBeingTrained] = [.1] * num_movies
	userValue = userFeature[featureBeingTrained]
	movieValue = movieFeature[featureBeingTrained]

	# Traverse input and train for this particular feature
	input_data = open('all.dta', 'r')
	l = 0
	for line in input_data:
		line = line.strip()
		user, movie, date, rating = line.split(' ')
		user, movie, date, rating = int(user) - 1, int(movie) - 1, int(date), int(rating)

		# Train on this data point a few times.
		# TODO: switch this to: continue until error is less than ...
		i = 0
		while i < num_iter:
			err = lrate * (rating - predict_rating(movie, user))
			uv = userValue[user]
			userValue[user] += err * movieValue[movie]
			movieValue[movie] += err * uv
			i += 1

		if l % 100 == 0:
			print l
		l += 1
	input_data.close()

	print featureBeingTrained, 'done'

# Done! Write the userFeature and movieFeature arrays out via pickling
userFeature_output = open('userFeature', 'wb')
pickle.dump(userFeature, userFeature_output)
userFeature_output.close()
movieFeature_output = open('movieFeature', 'wb')
pickle.dump(movieFeature, movieFeature_output)
movieFeature_output.close()


