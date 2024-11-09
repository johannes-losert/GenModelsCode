import pickle

# Replace 'your_file.pkl' with the actual filename of your .pkl file
file_name = "20241109_002033_cardratings_dict.pkl"

# Open and load the dictionary from the pickle file
with open(file_name, 'rb') as pkl_file:
    data = pickle.load(pkl_file)

# Display the loaded data
print(data)
