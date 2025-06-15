import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
from tkinter import ttk  # Para la tabla


# Ruta al modelo
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "blackbox_S.keras")
model = load_model(model_path)

def predict_point(model, x1, x2):
    data = np.array([[x1, x2]])
    prediction = model.predict(data, verbose=0)[0][0]
    return int(prediction > 0.5), prediction

def predict_batch(model, x1_list, x2_list):
    data = np.column_stack((x1_list, x2_list))
    predictions = model.predict(data, verbose=0)
    return (predictions > 0.5).astype(int).flatten().tolist()

# Ventana principal con menú
def mostrar_menu():
    menu = tk.Tk()
    menu.title("Menú - Black Box")
    menu.geometry("600x500")

    # Fondo con imagen
    try:
        img = Image.open("fondo.jpg")  # Asegúrate de tener esta imagen
        fondo = ImageTk.PhotoImage(img.resize((600, 500)))
        label_fondo = tk.Label(menu, image=fondo)
        label_fondo.place(x=0, y=0, relwidth=1, relheight=1)
        label_fondo.image = fondo
    except:
        menu.configure(bg="#DDEEFF")

    def abrir_aplicacion():
        menu.destroy()
        lanzar_clasificador()

    def mostrar_mensaje(titulo, texto):
        messagebox.showinfo(titulo, texto)
    
    # Título del menú
    tk.Label(menu, text="Menú Principal - Black Box", font=("Arial", 16, "bold"), bg="#DDEEFF", fg="#333").pack(pady=20)


    # Botones del menú principal
    tk.Button(menu, text="Iniciar Aplicación", font=("Arial", 12, "bold"), bg="#007ACC", fg="white", command=abrir_aplicacion, width=30).pack(pady=12)

    tk.Button(menu, text="¿Qué es Black Box?", font=("Arial", 12), command=lambda: mostrar_mensaje("¿Qué es Black Box?", "Black Box es un modelo entrenado que clasifica puntos en 2D (x₁, x₂) en clases binarias usando una red neuronal."), width=30).pack(pady=8)

    tk.Button(menu, text="Información de la Aplicación", font=("Arial", 12), command=lambda: mostrar_mensaje("Información", "Aplicación interactiva que predice visualmente la clase de un punto en un plano bidimensional y muestra la frontera de decisión aprendida."), width=30).pack(pady=8)

    tk.Button(menu, text="Créditos", font=("Arial", 12), command=lambda: mostrar_mensaje("Créditos", "Desarrollado por: Tu Nombre\nUniversidad XYZ - 2025"), width=30).pack(pady=8)

   
    
    tk.Button(menu, text="Salir", command=menu.quit, font=("Arial", 11), bg="red", fg="white", width=15).pack(pady=10)

    menu.mainloop()
# Lista global para almacenar el historial
historial = []

# VENTANA SECUNDARIA
def mostrar_tabla_resultados():
    ventana_tabla = tk.Toplevel()
    ventana_tabla.title("Tabla de Resultados")
    ventana_tabla.geometry("500x400")
    ventana_tabla.configure(bg="#F2F2F2")

    tk.Label(ventana_tabla, text="Historial de Predicciones", font=("Arial", 14, "bold"), bg="#F2F2F2").pack(pady=10)

    tabla = ttk.Treeview(ventana_tabla, columns=("x1", "x2", "clase", "prob"), show="headings")
    tabla.heading("x1", text="x₁")
    tabla.heading("x2", text="x₂")
    tabla.heading("clase", text="Clase")
    tabla.heading("prob", text="Probabilidad")

    for dato in historial:
        tabla.insert('', 'end', values=dato)

    tabla.pack(pady=10)

    explicacion = (
        "Esta tabla almacena los valores ingresados (x₁, x₂), la clase predicha por el modelo\n"
        "y la probabilidad obtenida. El modelo es una red neuronal que clasifica los puntos\n"
        "en 2D en dos clases (0 o 1), dependiendo de su ubicación relativa en el plano."
    )

    tk.Label(ventana_tabla, text=explicacion, bg="#F2F2F2", fg="#444", font=("Arial", 10), justify="left").pack(padx=10, pady=10)

# VENTANA PRINCIPAL
def lanzar_clasificador():
    root = tk.Tk()
    root.title("Clasificador con Modelo Keras")
    root.geometry("1000x600")
    root.configure(bg="#F2F2F2")

    # FRAME IZQUIERDO (SECCIÓN 2) - Entrada de datos y botones
    frame_izquierda = tk.Frame(root, bg="#F2F2F2", padx=20, pady=20)
    frame_izquierda.pack(side="left", fill="y")

    tk.Label(frame_izquierda, text="Ingrese x₁:", font=("Arial", 12), bg="#F2F2F2").pack(pady=(5, 0))
    entry_x1 = tk.Entry(frame_izquierda, font=("Arial", 12))
    entry_x1.pack()

    tk.Label(frame_izquierda, text="Ingrese x₂:", font=("Arial", 12), bg="#F2F2F2").pack(pady=(10, 0))
    entry_x2 = tk.Entry(frame_izquierda, font=("Arial", 12))
    entry_x2.pack()

    label_resultado = tk.Label(frame_izquierda, text="Predicción del modelo:", font=("Arial", 12, "bold"), bg="#F2F2F2")
    label_resultado.pack(pady=10)

    def predecir():
        try:
            x1 = float(entry_x1.get())
            x2 = float(entry_x2.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos.")
            return

        clase, probabilidad = predict_point(model, x1, x2)
        label_resultado.config(text=f"Predicción: Clase {clase} (Prob: {probabilidad:.2f})")

        # Guardar en historial
        historial.append((f"{x1:.2f}", f"{x2:.2f}", str(clase), f"{probabilidad:.2f}"))

        # Generar gráfica
        fig, ax = plt.subplots(figsize=(10, 10))
        x1_vals = np.linspace(-0.50, 2.0, 100)
        x2_vals = np.linspace(-0.50, 2.0, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = predict_batch(model, X1.ravel(), X2.ravel())
        Z = np.array(Z).reshape(X1.shape)

        ax.contourf(X1, X2, Z, levels=[-0.1, 0.5, 1.1], cmap="coolwarm", alpha=0.8)
        ax.plot(x1, x2, 'ko', label='Punto ingresado')
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("Mapa de predicción del modelo")
        ax.grid(True)
        ax.legend()

        for widget in frame_derecha.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_derecha)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    def limpiar():
        entry_x1.delete(0, tk.END)
        entry_x2.delete(0, tk.END)
        label_resultado.config(text="Predicción del modelo:")

    def volver_menu():
        root.destroy()
        mostrar_menu()

    tk.Button(frame_izquierda, text="Predecir y Graficar", command=predecir, bg="#007ACC", fg="white", font=("Arial", 12)).pack(pady=10)
    tk.Button(frame_izquierda, text="Limpiar Entradas", command=limpiar, bg="orange", font=("Arial", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Ver Tabla de Resultados", command=mostrar_tabla_resultados, bg="green", fg="white", font=("Arial", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Volver al Menú", command=volver_menu, bg="gray", fg="white", font=("Arial", 11)).pack(pady=5)

    # FRAME DERECHO (SECCIÓN 1) - Gráfica
    frame_derecha = tk.Frame(root, bg="#F2F2F2", padx=10, pady=10)
    frame_derecha.pack(side="right", fill="both", expand=True)

    root.mainloop()
# Iniciar menú principal
mostrar_menu()
