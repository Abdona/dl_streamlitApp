import streamlit as st
#from keras.layers import *
#import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import time as tt
#import io
#from contextlib import redirect_stdout
import pandas as pd
from scikeras.wrappers import KerasClassifier

#st.title('Deep Learning Assignment')

class StreamlitCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.epoch = 0
        self.progress_bar = st.progress(0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        st.text(f"Epoch {self.epoch}/{self.total_epochs}")

    def on_batch_end(self, batch, logs=None):
        current = ((self.epoch - 1) * self.total_batches + batch + 1)
        total = self.total_epochs * self.total_batches
        progress = current / total
        self.progress_bar.progress(progress)
        # Optionally show loss/accuracy
        #st.text(f"Loss: {logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        st.text("Training completed! ✅")
        self.progress_bar.progress(1.0)

tab1, tab2, tab3 = st.tabs(['Plot Data','Train', 'Predict'])
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

###scaling
x_train = x_train/255
x_test = x_test/255

### select class label to show first occurence of each class image
with tab1:
    selected_class_label = st.selectbox('Select class label', options = [0,1,2,3,4,5,6,7,8,9])
    class_index = np.where(y_train == selected_class_label)
    st.image(x_train[class_index[0][0]], width = 300, caption = f'Class label : {y_train[class_index[0][0]]}', use_container_width = True)

#### Build and train model with selected parameters
with tab2:
    apply_grid_search = st.toggle('Apply GridSearch', value = False)

    if not apply_grid_search:
        if st.toggle('Build Default Model'):
            ### Sequential model building
            nn_model = tf.keras.Sequential([
            Flatten(input_shape=(28,28)),
            Dense(units=512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
            
            nn_model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics  = ['accuracy'])

            summary, train, load_model, evaluate, reset = st.columns(5)

            evalute_tooltip = None
            #nn_model_history = None

            st.session_state['evaluate_button_disable'] = False

            if 'model_history' not in st.session_state or st.session_state['model_history'] == None:
                st.session_state['model_history'] = None
                st.session_state['evaluate_button_disable'] = True

            if st.session_state['evaluate_button_disable'] == True:
                evalute_tooltip = 'Model Should be Trained before Evaluated :)'

            if summary.button('Model Summary', use_container_width = True):
                st.write(nn_model.summary(print_fn = lambda x: st.text(x)))

            if train.button('Model Train', use_container_width = True):
                st.write('Model Train Started')
                nn_model_history = nn_model.fit(x=x_train, y=y_train, validation_split = 0.2, batch_size= 32 , epochs= 5, callbacks = [StreamlitCallback(5,len(x_train) // 32)], verbose = 0)
                st.session_state['model_history'] = nn_model_history
                st.session_state['evaluate_button_disable'] = False
                #st.write(nn_model.evaluate(x_train, y_train))
                #nn_model.save(filepath = '/Users/abdulrahmannaser/Downloads/model_streamlit.h5')
                    
            if evaluate.button('Model Evaluate', use_container_width = True, disabled = st.session_state['evaluate_button_disable'], help = evalute_tooltip):
                st.write(st.session_state['model_history'].model.evaluate(x_train, y_train))

            if load_model.button('Load Model', use_container_width = True, disabled = False):
                nn_model_loaded = tf.keras.models.load_model('/Users/abdulrahmannaser/Downloads/model_streamlit.h5', compile = False)
                nn_model_loaded.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics  = ['Accuracy'])
                st.write(nn_model_loaded.evaluate(x_train, y_train))
            
            if reset.button('Reset', use_container_width = True):
                st.session_state.clear()


        #### Build model from scratch by selecting the layers
        else:
            if 'sequential' not in st.session_state or st.session_state['sequential'] == None:
                st.session_state['sequential'] = tf.keras.Sequential()
            
            if 'dense_button' not in st.session_state:
                st.session_state['dense_button'] = False
            
            if 'flatten_button' not in st.session_state:
                st.session_state['flatten_button'] = False

            if 'summary_button' not in st.session_state:
                st.session_state['summary_button'] = False
            
            if 'reset_button' not in st.session_state:
                st.session_state['reset_button'] = False
            
            if 'train_button' not in st.session_state:
                st.session_state['train_button'] = False
            
            st.write('Select / Reset Layers')
            train, dense, flatten, summary, reset = st.columns(5)

            if dense.button('Dense Layer', use_container_width = True):
                st.session_state['dense_button'] = True
                st.session_state['flatten_button'] = False
                st.session_state['summary_button'] = False
                st.session_state['reset_button'] = False
                st.session_state['train_button'] = False

            if flatten.button('Flatten Layer',use_container_width = True):
                st.session_state['flatten_button'] = True
                st.session_state['dense_button'] = False
                st.session_state['summary_button'] = False
                st.session_state['reset_button'] = False
                st.session_state['train_button'] = False
            
            if summary.button('Summary',use_container_width = True):
                st.session_state['summary_button'] = True
                st.session_state['flatten_button'] = False
                st.session_state['dense_button'] = False
                st.session_state['reset_button'] = False
                st.session_state['train_button'] = False
            
            if reset.button('Reset',use_container_width = True):
                st.session_state['reset_button'] = True
                st.session_state['summary_button'] = False
                st.session_state['flatten_button'] = False
                st.session_state['dense_button'] = False
                st.session_state['train_button'] = False

            if train.button('Train', use_container_width = True):
                st.session_state['train_button'] = True
                st.session_state['reset_button'] = False
                st.session_state['summary_button'] = False
                st.session_state['flatten_button'] = False
                st.session_state['dense_button'] = False
            #reset_bt = reset.button('Reset',use_container_width = True)

            if st.session_state['dense_button']:
                name = 'Dense' + str(len(st.session_state['sequential'].layers))
                units_number_col , activation_function_col = st.columns(2)
                add_layer = st.button('Add Dense')
                units_number = units_number_col.radio('Number of Units', [10,32,64,128,256,512], index = 1)
                activation_function = activation_function_col.radio('Activation Function', ['relu','sigmoid','tanh','softmax'])
                if add_layer:
                    st.session_state['sequential'].add(Dense(units_number, activation_function, name=name))
                    st.write(st.session_state['sequential'].summary(print_fn = lambda x : st.text(x)))
            
            if st.session_state['flatten_button']:
                input_shape = x_train[0].shape
                name = 'Flatten' + str(len(st.session_state['sequential'].layers))
                add_layer = st.button('Add Flatten')
                if add_layer:
                    st.session_state['sequential'].add(Flatten(input_shape = input_shape, name=name))
                    st.write(st.session_state['sequential'].summary(print_fn = lambda x : st.text(x)))
            
            if st.session_state['summary_button']:
                st.write(st.session_state['sequential'].summary(print_fn = lambda x : st.text(x)))
           
            if st.session_state['reset_button']:
               st.session_state['sequential'] = None
            
            if st.session_state['train_button']:
                if 'rmsprop' not in st.session_state:
                    st.session_state['rmsprop'] = None

                if 'nesterov' not in st.session_state:
                    st.session_state['nesterov'] = None

                if 'mgd' not in st.session_state:
                    st.session_state['mgd'] = None

                if 'sgd' not in st.session_state:
                    st.session_state['sgd'] = None

                
                if 'adam' not in st.session_state:
                    st.session_state['adam'] = None
                
                if 'performance_df' not in st.session_state:
                    st.session_state['performance_df'] = {'optimizer':[], 'accuracy_train':[], 'loss_train':[],'accuracy_test':[], 'loss_test':[],'execution_time':[]}

                spilting_col, optimizer_col , batch_size_col, epochs_col = st.columns(4)
                spilting = spilting_col.toggle('Validation Split')
                optimizer = optimizer_col.radio('Select Optimizer', ['adam','rmsprop', 'nesterov', 'sgd', 'mgd'])
                batch_size = batch_size_col.radio('Batch Size', [32,64, 128, 256])
                epochs = epochs_col.radio('Epochs', [5, 10, 15, 20, 25, 30])


                if spilting:
                    valid_data_ration = spilting_col.slider('Split Ratio', 0.1,0.9, 0.2)
                    x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size= valid_data_ration)

                if optimizer == 'nesterov':
                    st.session_state['sequential'].compile(optimizer = keras.optimizers.SGD(nesterov = True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                if optimizer == 'mgd':
                    st.session_state['sequential'].compile(optimizer = keras.optimizers.SGD(momentum = 0.2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                if optimizer == 'sgd':
                    st.session_state['sequential'].compile(optimizer = 'sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                if optimizer == 'rmsprop':
                    st.session_state['sequential'].compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                if optimizer == 'adam':
                    st.session_state['sequential'].compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                
                if st.button('Start'):
                    start_time = tt.time()
                    if not spilting:
                        st.session_state['sequential'].fit(x=x_train, y=y_train, validation_split = 0.2, batch_size= batch_size , epochs= epochs, callbacks = [StreamlitCallback(epochs,len(x_train) // batch_size)], verbose = 0)
                    else:
                        st.session_state['sequential'].fit(x=x_train, y=y_train, validation_data = (x_val,y_val), batch_size= batch_size , epochs= epochs, callbacks = [StreamlitCallback(epochs,len(x_train) // batch_size)], verbose = 0)
                    end_time = tt.time()

                    execution_time = end_time - start_time
                    accuracy_train = st.session_state['sequential'].evaluate(x_train,y_train)
                    accuracy_test = st.session_state['sequential'].evaluate(x_test,y_test)

                    st.session_state[optimizer] = {'model': st.session_state['sequential'], 'execution_time': execution_time, 'accuracy_train' : accuracy_train, 'accuracy_test' : accuracy_test}
                    
                    if optimizer in st.session_state['performance_df']['optimizer']:
                        index = st.session_state['performance_df']['optimizer'].index(optimizer)
                        st.session_state['performance_df']['accuracy_train'][index] = accuracy_train[1]
                        st.session_state['performance_df']['loss_train'][index] = accuracy_train[0]
                        st.session_state['performance_df']['accuracy_test'][index] = accuracy_test[1]
                        st.session_state['performance_df']['loss_test'][index] = accuracy_test[0]
                        st.session_state['performance_df']['execution_time'][index] = execution_time / 60
                    
                    else:
                        st.session_state['performance_df']['optimizer'].append(optimizer)
                        st.session_state['performance_df']['accuracy_train'].append(accuracy_train[1])
                        st.session_state['performance_df']['loss_train'].append(accuracy_train[0])
                        st.session_state['performance_df']['accuracy_test'].append(accuracy_test[1])
                        st.session_state['performance_df']['loss_test'].append(accuracy_test[0])
                        st.session_state['performance_df']['execution_time'].append(execution_time / 60)


                
                if st.button('Show Performance'):
                    st.table(pd.DataFrame(st.session_state['performance_df']))
    
    #### Grid Search Tab
    else:
        if 'model_hyper_tuning' not in st.session_state:
            st.session_state['model_hyper_tuning'] = None
        
        left_col, middle_col, right_col  = st.columns(3)

        epochs = left_col.pills('Epochs', [5,10], selection_mode = "multi", default = [5,10])
        hidden_layers = left_col.pills('Hidden Layers', [3,4,5], selection_mode = "multi", default = [3,4,5])
        units_numbers = left_col.pills('Units', [32,64,128], selection_mode = "multi", default = [32,64,128])
        l2s = left_col.pills('L2 regularization', [0, 0.0005, 0.5], selection_mode = "multi",default = [0,0.0005,0.5])
        learning_rates = middle_col.pills('Learning rate', [0.001,0.0001], selection_mode = "multi", default  = [0.001,0.0001])
        optimizers = middle_col.pills('Optimizers', ['nesterov', 'rmsprop', 'adam', 'momentum', 'sgd'], selection_mode = "multi", default = ['nesterov', 'rmsprop', 'adam', 'momentum', 'sgd'])
        batch_sizes = middle_col.pills('Batch size', [16,32,64], selection_mode = "multi", default = [16,32,64])
        activation_functions = middle_col.pills('Activation', ['sigmoid', 'tanh', 'relu'], selection_mode = "multi", default = ['sigmoid', 'tanh', 'relu'])

        droupout_toggle = right_col.toggle('Droupout')
        batch_norm_toggle = right_col.toggle('Batch Normalization')
        l1_l2_toggle = right_col.toggle('L1 and L2')

        optimizers_method = {
            'nesterov' :  keras.optimizers.SGD(nesterov = True),
            'rmsprop' : keras.optimizers.RMSprop,
            'adam': keras.optimizers.Adam,
            'momentum' : keras.optimizers.SGD(momentum = 0.1),
            'sgd': keras.optimizers.SGD
                          }
        
        hyper_parameters = {
            'model__batch_size' : batch_sizes, 
            'fit__epochs' : epochs, 
            'optimizer' : optimizers,
            'model__hidden_layer': hidden_layers,
            'model__activation': activation_functions,
            'optimizer__learning_rate': learning_rates,
            'model__units': units_numbers,
            'model__kernel_regularizer': l2s
            }
        
        def get_clf(hidden_layer, units, activation, **kwargs):
            model = keras.Sequential()
            model.add(Flatten(input_shape = x_train[0].shape))
            for i in range(hidden_layer):
                model.add(Dense(units = units, activation = activation))
            model.add(Dense(units= 10, activation = 'softmax'))

            return model
        
        KerasClassifier(model = get_clf, loss = 'sparse_categorical_crossentropy')
        
        model_hyper_tuning = GridSearchCV(KerasClassifier(model = get_clf, loss = 'sparse_categorical_crossentropy'),cv=2, hyper_parameters,error_score='raise')

        if st.button('Start'):
            model_hyper_tuning.fit(x_train,y_train)
            st.session_state['model_hyper_tuning'] = model_hyper_tuning

            st.write(st.session_state['model_hyper_tuning'].best_score_)

