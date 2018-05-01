#Code Written by Chirag Mahaveer Parmar
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

##################### DATA PREPARATION #############################
option = input("Do you want to view the graphs related to the data (yes - 1 / no - 2)")
#data retrieval
raw_df = pd.read_csv("./winequality-red.csv")

#seperate the output column from the dataset
quality = raw_df.quality
df = raw_df.drop("quality",axis = 1)
column_names = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11']

#calculate maximum value of every column and normalize the whole column
max_fixed_acidity = df.fixed_acidity/max(df.fixed_acidity)
max_volatile_acidity = df.volatile_acidity/max(df.volatile_acidity)
max_citric_acid = df.citric_acid/max(df.citric_acid)
max_residual_sugar = df.residual_sugar/max(df.residual_sugar)
max_chlorides = df.chlorides/max(df.chlorides)
max_free_sulfur_dioxide = df.free_sulfur_dioxide/max(df.free_sulfur_dioxide)
max_total_sulfur_dioxide = df.total_sulfur_dioxide/max(df.total_sulfur_dioxide)
max_density = df.density/max(df.density)
max_pH = df.pH/max(df.pH)
max_sulphates = df.sulphates/max(df.sulphates) 
max_alcohol =  df.alcohol/max(df.alcohol)

#calculate average value of every value nad normalize the whole column
avg_fixed_acidity = df.fixed_acidity/np.mean(df.fixed_acidity)
avg_volatile_acidity = df.volatile_acidity/np.mean(df.volatile_acidity)
avg_citric_acid = df.citric_acid/np.mean(df.citric_acid)
avg_residual_sugar = df.residual_sugar/np.mean(df.residual_sugar)
avg_chlorides = df.chlorides/np.mean(df.chlorides)
avg_free_sulfur_dioxide = df.free_sulfur_dioxide/np.mean(df.free_sulfur_dioxide)
avg_total_sulfur_dioxide = df.total_sulfur_dioxide/np.mean(df.total_sulfur_dioxide)
avg_density = df.density/np.mean(df.density)
avg_pH = df.pH/np.mean(df.pH)
avg_sulphates = df.sulphates/np.mean(df.sulphates) 
avg_alcohol =  df.alcohol/np.mean(df.alcohol)

#normalized dataset using the max of column method
max_df = pd.DataFrame({'fixed_acidity': max_fixed_acidity, 'volatile_acidity': max_volatile_acidity, 'citric_acid':max_citric_acid, \
	'residual_sugar': max_residual_sugar, 'chlorides': max_chlorides, 'free_sulfur_dioxide': max_free_sulfur_dioxide, \
	'total_sulfur_dioxide': max_total_sulfur_dioxide, 'density': max_density, 'pH': max_pH, 'sulphates': max_sulphates, 'alcohol':max_alcohol})

#normalized datset using the average of column method
avg_df = pd.DataFrame({'fixed_acidity': avg_fixed_acidity, 'volatile_acidity': avg_volatile_acidity, 'citric_acid':avg_citric_acid, \
	'residual_sugar': avg_residual_sugar, 'chlorides': avg_chlorides, 'free_sulfur_dioxide': avg_free_sulfur_dioxide, \
	'total_sulfur_dioxide': avg_total_sulfur_dioxide, 'density': avg_density, 'pH': avg_pH, 'sulphates': avg_sulphates, 'alcohol':avg_alcohol})

#principal component analysis of all three data frames
pca = PCA()
pca_df = pd.DataFrame(pca.fit_transform(df), columns = column_names)
pca_avg_df = pd.DataFrame(pca.fit_transform(avg_df), columns = column_names)
pca_max_df = pd.DataFrame(pca.fit_transform(max_df), columns = column_names)

######################### VIZ - SCREE CHARTS & SCATTER ###########################

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#008080", "#e6beff"]

if option == 1:
	plt.hist(quality, bins=10, color = PALETTE[7])
	plt.show()

def scree_it(df,title):
	std_df = df.std(axis=None, skipna=None, level=None, ddof=0)
	var_df = std_df**(2)
	cum_var_df = var_df.cumsum()
	prop_df = var_df/sum(var_df)
	scree_df = pd.DataFrame({"std" : std_df, "var": var_df, "cum_var": cum_var_df, "prop" : prop_df})
	x_axes = list(scree_df.index)
	#print(x_axes, scree_df.prop)
	plt.figure(1)
	plt.bar(x_axes, scree_df.prop, color= PALETTE, align='center')
	plt.ylabel('Proportion')
	plt.xlabel('Variables')
	plt.ylim(0,1)
	plt.title(title)
	plt.show()

if option == 1:
	scree_it(df, 'Raw Dataset')
	scree_it(avg_df, 'Normalized(Average)')
	scree_it(max_df, 'Normalized(Maximum)')
	scree_it(pca_df, 'PCA - Raw')
	scree_it(pca_avg_df, 'PCA - Normalized(Average)')
	scree_it(pca_max_df, 'PCA - Normalized(Maximum)')

#add the quality column back in every dataframe
df['quality'] = quality
avg_df['quality'] = quality
max_df['quality'] = quality
pca_df['quality'] = quality
pca_max_df['quality'] = quality
pca_avg_df['quality'] = quality

def scatter_it(df, title):
	column_ids = df.columns.values
	plt.figure(1)
	plt.title(title)
	for i in range(0,len(column_ids)-1):
		plt.subplot(2,6,i+1)
		plt.scatter(x=df[column_ids[i]], y=df[column_ids[(i+1) % 11]], c = df.quality, label = df.quality, alpha= 0.5)
		plt.xlabel(column_ids[i])
		plt.ylabel(column_ids[(i+1) % 11])
	plt.subplots_adjust(left=0.04, bottom=0.08, right=0.96, top=0.98, wspace=0.28, hspace=0.2)
	plt.show()

if option == 1:
	scatter_it(df, 'Raw Dataset')
	scatter_it(avg_df, 'Normalized(Average)')
	scatter_it(max_df, 'Normalized(Maximum)')
	scatter_it(pca_df, 'PCA - Raw')
	scatter_it(pca_avg_df, 'PCA - Normalized(Average)')
	scatter_it(pca_max_df, 'PCA - Normalized(Maximum)')

if option == 1:
	scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	scatter_matrix(avg_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	scatter_matrix(max_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	scatter_matrix(pca_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	scatter_matrix(pca_avg_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	scatter_matrix(pca_max_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
	plt.show()

#################### IMPLEMENT A MODEL ###########################

#break data into training and testing data
columns = df.columns.values
input_columns = columns[1:11]
df_train, df_test = train_test_split(df, test_size=0.2)
max_df_train, max_df_test = train_test_split(max_df, test_size=0.2)
avg_df_train, avg_df_test = train_test_split(avg_df, test_size=0.2)
pca_df_train, pca_df_test = train_test_split(pca_df, test_size=0.2)
pca_max_df_train, pca_max_df_test = train_test_split(pca_max_df, test_size=0.2)
pca_avg_df_train, pca_avg_df_test = train_test_split(pca_avg_df, test_size=0.2)

#a.k.a modelling "no pun intended"
def neuralnet_it(df_train, df_test, Input, Output):
	mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=50000)
	mlp.fit(df_train[Input],df_train[Output])
	predictions = mlp.predict(df_test[Input])
	return predictions, accuracy_score(df_test[Output], predictions)

df_predictions, df_acc = neuralnet_it(df_train, df_train, input_columns, "quality")
avg_df_predictions, avg_df_acc = neuralnet_it(max_df_train, max_df_test, input_columns, "quality")
max_df_predictions, max_df_acc = neuralnet_it(avg_df_train, avg_df_test, input_columns, "quality")
pca_df_predictions, pca_df_acc = neuralnet_it(pca_df_train, pca_df_test, column_names, 'quality')
pca_max_df_predictions, pca_max_df_acc = neuralnet_it(pca_max_df_train, pca_max_df_test, column_names, 'quality')
pca_avg_df_predictions, pca_avg_df_acc = neuralnet_it(pca_avg_df_train, pca_avg_df_test, column_names, 'quality')

print("Raw Dataset Accuracy: ", df_acc)
print("Normalized(average) Dataset Accuracy: ", avg_df_acc)
print("Normalized(max) Dataset Accuracy: ", max_df_acc)
print("PCA(raw) Dataset Accuracy: ", pca_df_acc)
print("PCA(normalized average) Dataset Accuracy: ", pca_max_df_acc)
print("PCA(normalized maximum) Dataset Accuracy: ", pca_avg_df_acc)

