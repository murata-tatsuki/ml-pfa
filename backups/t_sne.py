import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.python.keras.models import Model
from Networks.Tools import datatools, modeltools, evaltools
from Networks.PairModel import models, training


if __name__ == "__main__":
    print("Pair Model Evaluation...")
    print("T-SNE...")
    
    model_name = "Pair_Model_Standard_201129_Standard_loss_weights_vertex0.1_position0.9_1000epochs_plus_loss_weights_vertex0.9_position0.1_1500epochs_plus_loss_weights_vertex0.95_position0.05_500epochs_lr_0.0001"
    data_path = "data/numpy/Pair_evaluating_reshaped.npy"

    print("Data Loading ...")
    data = np.load(data_path)
    variables, vertex, position = datatools.GetPairData(data, Cov=True)
    index = np.random.permutation(len(variables))
    x = variables[index][:10000]
    y = vertex[index][:10000]
    my = np.argmax(y, axis=1)

    print("Model Loading ...")
    model = modeltools.LoadPairModel(model_name)

    model.compile(loss={'Vertex_Output': 'categorical_crossentropy', 'Position_Output': 'mean_squared_logarithmic_error'},
                  optimizer='SGD',
                  metrics=['accuracy', 'mae'])

    classes = ["NC", "PV", "SVCC", "SVBB", "TVCC", "SVBC", "Others"]
    colors = ["dimgrey", "red", "orange", "blue", "yellowgreen", "darkgreen", "purple"]
    ites = [500, 1000, 5000, 10000]
    
    new_model = Model(inputs=model.input, outputs=model.get_layer("Activation_ReLU_3").output)
    middle = new_model.predict([x])

    for ite in ites:
        tsne = TSNE(n_components=2, random_state = 0, perplexity = 100, n_iter = ite)

        middle_embedded = tsne.fit_transform(middle)
        x_embedded = tsne.fit_transform(x)

        for i, (col, cla) in enumerate(zip(colors, classes)):
            tmp = [xtmp for xtmp, ytmp in zip(middle_embedded, my) if ytmp == i]
            tmp = np.array(tmp)
            plt.scatter(tmp[:, 0], tmp[:, 1], label=cla, color=col, s=5)
        plt.legend(loc='upper left')
        plt.savefig("data/figure/tsne/T_SNE_Pair_Model_Standard_Middele_rs0_per100_ite" + str(ite) + "_r10000sample.pdf")
        plt.cla()

        for i, (col, cla) in enumerate(zip(colors, classes)):
            tmp = [xtmp for xtmp, ytmp in zip(x_embedded, my) if ytmp == i]
            tmp = np.array(tmp)
            plt.scatter(tmp[:, 0], tmp[:, 1], label=cla, color=col, s=5)
        plt.legend(loc='upper left')
        plt.savefig("data/figure/tsne/T_SNE_Pair_Model_Standard_rs0_per100_ite" + str(ite) + "_r10000sample.pdf")
        plt.cla()
