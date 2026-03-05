import cv2
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils_med import CustomDS_Med

current_dir= os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'PetImages')


def gradCam(model, device, target_class:int, datos:dict, data_path:str):
    #Primero lo ponemss en evalcuación
    model.to(device)
    model.eval()
    activaciones, gradientes = None, None
    
    Data_to_Steal = {"Activations":None, "Gradients":None}
    #Definimos nuestros ganchos a la capa convolucional para espiar y robar las activaciones, así como los gradientes de los mapas de activación.
    def hook_activaciones(modulo, input, output):
        Data_to_Steal["Activations"] = output.detach()

    def hook_gradientes(modulo, input, output):
        Data_to_Steal['Gradients'] = output[0].detach()

    #Ahora tenemos que obtener la capa objetivo
    target_layer = model.layer4[-1]
    ActivationHook = target_layer.register_forward_hook(hook_activaciones)
    GradientHook = target_layer.register_backward_hook(hook_gradientes)

    #Ahora hacemos una predicción con una imagen de gato
    GANGTEL = pd.DataFrame(datos)
    RandomDataSet = CustomDS_Med(dataframe=GANGTEL, images_dir=data_path, transform=None)
    image_DataLoader = DataLoader(dataset=RandomDataSet, batch_size=1)

    data_iter = iter(image_DataLoader)
    image, label, filename = next(data_iter)

    image = image.to(device)
    label = label.to(device)

    print(f"La imagen está en {image.device} y la etiqueta en {label.device}\n Nota: Si es cuda:0  es que está en GPU, sino está en la CPU ")

    #Ahora hacemos una predicción con esa imagen:
    prediccion = model(image) #Aquí se roban las activaciones gracias a los "espias" o hooks

    #Ahora para robar los gradientes de la capa convolucional 8

    """Primero definimos lo que será la capa de salida, para hacer el backprop"""
    target_score = prediccion[0, target_class] #Como solamente le pasamos un solo ejemplo, la matriz de salida será de la forma 1 x 2, donde las filas son los ejemplos y las columnas los logits de cada clase [0.3, 0.13l4]

    model.zero_grad() #Calculo de gradientes
    target_score.backward(retain_graph=True) #Ahora calculamos los gradientes con respecto a la neurona objetivo (la de gato), si quisieras ver el por qué decidió que era perro o así, cambia la neurona de salida

    #Ahora ya que robamos los gradientes y activaciones podemos hacer lo que menciona el paper de grad-CAM v2 https://arxiv.org/pdf/1610.02391
    G = Data_to_Steal['Gradients']
    A = Data_to_Steal['Activations']

    #Estps son los pesos de las activaciones, dando un tensor [1, 128, 1, 1]
    pesos = torch.mean(G, dim = [2, 3], keepdim=True)
    print(f"Forma de los pesos: {pesos.shape}")

    #Ahora que tenemos los pesos, debemos saber que tanto influyeron en las activaciones.
    PesosA = A * pesos
    print(PesosA.shape)

    #Ahora hacemos la suma ponderada de los pesos de cada activación
    heatmap = torch.sum(PesosA, dim=1)
    print(heatmap.shape)

    #Ahora aplicamos ReLu porque solo queremos lo positivo, lo negativo no nos interesa para nada.
    heatmap = torch.nn.functional.relu(heatmap)

    #Ahora normalizamos entre -1 y 1
    print(heatmap.max())

    heatmap -= heatmap.min()
    heatmap /= heatmap.max()

    #Ahora lo metemos todo a CPU, así como quitar el batch para hacerlo un arreglo de NumPy
    print(f"Antes de ponerlo chulo: {heatmap.shape}")
    heatmap = heatmap.squeeze(0).cpu().numpy()
    print(f"Después de arreglarlo un poco xd: {heatmap.shape}")

    #Ahora debemos hacer todo lo de mostrar la imagen con Cv2 bruh
    label = GANGTEL.iloc[0, 1] #Accedo a la fila del indice 0 y obtengo el elemento de la segunda columna "label"
    print(f"Etiqueta: {label}, Nombre: {filename[0]}") #Nomás pa' checar jaja, pero como vimos hace un rato hicimos una lista, por lo que es una lista, así que accedo a el elemento con [0]
    img = cv2.imread(rf'{data_path}\{label}\{filename[0]}') #Ahora que tenemos la ruta, leemos la imagen
    img = cv2.resize(img, (128, 128)) #La ajustamos al tamaño 128, 128

    #Acá también ajustamos el mapa de calor xd
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    #Superponemos el mapa de calor sobre la imagen
    img = (heatmap * 0.4) + img
    img = np.uint8(img)

    NumArchivo = filename[0] #"10.jpg"
    NumArchivo = NumArchivo.split(".") #['10', '.jpg']
    NumArchivo = NumArchivo[0] #10
    print(NumArchivo)
    cv2.imwrite(fr'C:\Users\PC\Desktop\Abbadon\source\GradCAMS\GradCAM{NumArchivo}_{label}.jpg', img)

    ActivationHook.remove()
    GradientHook.remove()

    print(rf"La imagen se ha guardado en la ruta: C:\Users\PC\Desktop\Abbadon\source\GradCAMS\GradCAM{NumArchivo}_{label}.jpg")