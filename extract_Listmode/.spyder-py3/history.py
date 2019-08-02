
valid = pd.DataFrame()
for i in range(int(len(files_titles)*2/3), len(files_titles)*2/3):
    df = pd.read_csv(data_destination + '/features/' + files_titles[i], sep = ',', engine = 'python')
    valid = df.append(valid)
for i in range(int(len(files_titles)*2/3), len(files_titles)):
    df = pd.read_csv(data_destination + '/features/' + files_titles[i], sep = ',', engine = 'python')
    valid = df.append(valid)
valid.set_index(['Particle ID', 'date'], inplace = True)
valid = valid.dropna(how = 'any')

X_valid = valid.iloc[:, :-1]
y_valid = valid.iloc[:, -1]

# Label Encoding: Turns the labels into numbers
y_valid = le.transform(y_valid)

y_pred_valid = clf.predict(X_valid)
# evaluate predictions
accuracy = accuracy_score(y_valid, y_pred_valid)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_valid, y_pred_valid, target_names=le.classes_))
from sklearn.model_selection import GridSearchCV
X_train, y_train = rus.fit_sample(X,y)
rf = RandomForestClassifier(random_state=0, n_jobs = -1)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
best_rf = RandomForestClassifier(**CV_rfc.best_params)
best_rf = RandomForestClassifier(**CV_rfc.best_params_)
best_rf.fit(X_valid, y_valid)
best_rf.fit(X_train, y_train)
best_rf.pred
best_rf.predict
y_pred_valid = best_rf.predict(X_valid, y_valid)
y_pred_valid = best_rf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred_valid)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_valid, y_pred_valid, target_names=le.classes_))
import os 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


#from xgboost import XGBClassifier


#from keras.utils import plot_model

# The directory in which you have placed the following code
os.chdir('W:/Bureau/these/planktonPipeline')

from from_cytoclus_to_imgs import extract_imgs
from from_imgs_to_keras import imgs_train_test_valid_split, nb_available_imgs
from img_recognition import toy_model, toy_model2, plot_losses, multi_input_gen, fit_gen



# Where to look the data at and where to write treated data: Change with yours
data_source = 'W:/Bureau/these/donnees_oceano/new_process_FLR6'
data_destination = 'W:/Bureau/these/data'

seed = 42
model = toy_model()
#model.summary()
#plot_model(model, to_file='toy_model.png')

# For the moment the generator only output existing images. But rotated one etc can be generated
source_generator = ImageDataGenerator(horizontal_flip = True, rescale = 1./ 255) # Allow 180° rotations # rescale ? 
batch_size = 128
os.chdir(root_dir)

train_generator_dict = multi_input_gen(source_generator, 'train', batch_size, shuffle = True)
test_generator_dict = multi_input_gen(source_generator, 'test', batch_size, shuffle = True)
valid_generator_dict = multi_input_gen(source_generator, 'valid', batch_size, shuffle = True)

train_generator = fit_gen(train_generator_dict)
test_generator = fit_gen(test_generator_dict)
valid_generator = fit_gen(valid_generator_dict)


# Defining the number of steps in an epoch
nb_train, nb_test, nb_valid = nb_available_imgs(root_dir) # Compute the available images for each train, test and valid folder

STEP_SIZE_TRAIN = (nb_train // batch_size) + 1 
STEP_SIZE_VALID = (nb_valid // batch_size) + 1 
STEP_SIZE_TEST = (nb_test // batch_size) + 1 

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs = 2
)


# General picture of the valid and train losses through the epochs:
plot_losses(history)