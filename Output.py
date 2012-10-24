import pickle

# Variables and constants
num_features = 1


def predict_rating(movie, user):
    rating_sum = 0
    for i in range(num_features):
        rating_sum += userFeature[i][user] * movieFeature[i][movie]

    if rating_sum > 5:
        rating_sum = 5
    elif rating_sum < 1:
        rating_sum = 1
        
    return rating_sum


# Load movieFeature and userFeature
read_file = open('movieFeature', 'rb')
movieFeature = pickle.load(read_file)
read_file.close()
read_file = open('userFeature', 'rb')
userFeature = pickle.load(read_file)
read_file.close()

# Read lines from 'mu/qual.dta' write predicted ratings into 'solution'
input_data = open('mu/qual.dta', 'r')
output_data = open('solution', 'w')
for line in input_data:
    line = line.strip()
    user, movie, date = line.split(' ')
    user, movie, date = int(user) - 1, int(movie) - 1, int(date)
    rating = '%.3f' % predict_rating(movie, user)
    output_data.write(rating + '\n')
input_data.close()
output_data.close()
