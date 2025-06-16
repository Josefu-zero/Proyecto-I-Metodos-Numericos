import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
import ctypes
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

    # Centrar la ventana en la pantalla
    menu.update_idletasks()
    menu.title("Menú - Black Box")
    ancho_ventana = 600
    alto_ventana = 500
    ancho_pantalla = menu.winfo_screenwidth()
    alto_pantalla = menu.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    menu.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Fondo con imagen
    try:
    # Usar imagen de fondo desde la carpeta assets
        img_path = os.path.join(script_dir, "assets", "fondo2.png")
        img = Image.open(img_path)
        fondo = ImageTk.PhotoImage(img.resize((600, 500)))
        label_fondo = tk.Label(menu, image=fondo)
        label_fondo.place(x=0, y=0, relwidth=1, relheight=1)
        label_fondo.image = fondo
    except:
        menu.configure(bg="#2F2F2F")

    def abrir_aplicacion():
        menu.destroy()
        lanzar_clasificador()

    def mostrar_mensaje(titulo, texto):
        messagebox.showinfo(titulo, texto)
    
    # Título del menú
    tk.Label(menu, text="Menú Principal - Black Box", font=("Helvetica", 16, "bold"), bg="#DDEEFF", fg="#333").pack(pady=20)


    # Botones del menú principal
    tk.Button(menu, text="Iniciar Aplicación", font=("Helvetica", 12, "bold"), bg="#007ACC", fg="white", command=abrir_aplicacion, width=30).pack(pady=12)

    tk.Button(menu, text="¿Qué es Black Box?", font=("Helvetica", 12), command=lambda: mostrar_mensaje("¿Qué es Black Box?", "Black Box es un modelo entrenado que clasifica puntos en 2D (x₁, x₂) en clases binarias usando una red neuronal."), width=30).pack(pady=8)

    tk.Button(menu, text="Información de la Aplicación", font=("Helvetica", 12), command=lambda: mostrar_mensaje("Información", "Aplicación interactiva que predice visualmente la clase de un punto en un plano bidimensional y muestra la frontera de decisión aprendida."), width=30).pack(pady=8)

    tk.Button(menu, text="Créditos", font=("Helvetica", 12), command=lambda: mostrar_mensaje("Créditos", "Desarrollado por: Dany, Kevin, Joseph \nEscuela Poliecnica Nacional - 2025"), width=30).pack(pady=8)
   
    tk.Button(menu, text="Salir", command=menu.quit, font=("Helvetica", 11), bg="red", fg="white", width=15).pack(pady=10)

    menu.mainloop()
# Lista global para almacenar el historial
historial = []

# VENTANA SECUNDARIA
def mostrar_tabla_resultados():
    ventana_tabla = tk.Toplevel()
    ventana_tabla.title("Tabla de Resultados")
    ventana_tabla.geometry("500x400")
    ventana_tabla.configure(bg="#2F2F2F")
    # Centrar la ventana en la pantalla
    ventana_tabla.update_idletasks()
    ancho_ventana = 500
    alto_ventana = 400
    ancho_pantalla = ventana_tabla.winfo_screenwidth()
    alto_pantalla = ventana_tabla.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    ventana_tabla.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    tk.Label(ventana_tabla, text="Historial de Predicciones", font=("Helvetica", 14, "bold"), bg="#F2F2F2").pack(pady=10)

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

    tk.Label(ventana_tabla, text=explicacion, bg="#F2F2F2", fg="#444", font=("Helvetica", 10), justify="left").pack(padx=10, pady=10)

# VENTANA PRINCIPAL
def lanzar_clasificador():
    root = tk.Tk()
    root.title("Clasificador con Modelo Keras")
    root.configure(bg="#FFFFFF")
    # Centrar la ventana en la pantalla
    root.update_idletasks()
    ancho_ventana = 1000
    alto_ventana = 600
    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
    # FRAME IZQUIERDO (SECCIÓN 2) - Entrada de datos y botones
    frame_izquierda = tk.Frame(root, bg="#404040", padx=10, pady=10)
    # Imagen de fondo para el frame izquierdo
    try:
        bg_img_path = os.path.join(script_dir, "assets", "left_bg.png")
        bg_img = Image.open(bg_img_path)
        bg_img = bg_img.resize((300, 700))
        frame_izquierda.config(width=300, height=700)
        frame_izquierda.pack_propagate(False)
        bg_photo = ImageTk.PhotoImage(bg_img)
        bg_label = tk.Label(frame_izquierda, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo
    except Exception:
        frame_izquierda.configure(bg="#404040")
    frame_izquierda.pack(side="left", fill="y")

    tk.Label(frame_izquierda, text="Ingrese x₁:", font=("Helvetica", 12), bg="#F2F2F2").pack(pady=(5, 0))
    entry_x1 = tk.Entry(frame_izquierda, font=("Helvetica", 12))
    entry_x1.pack()

    tk.Label(frame_izquierda, text="Ingrese x₂:", font=("Helvetica", 12), bg="#F2F2F2").pack(pady=(10, 0))
    entry_x2 = tk.Entry(frame_izquierda, font=("Helvetica", 12))
    entry_x2.pack()

    label_resultado = tk.Label(frame_izquierda, text="Predicción del modelo:", font=("Helvetica", 12, "bold"), bg="#F2F2F2")
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

    tk.Button(frame_izquierda, text="Predecir y Graficar", command=predecir, bg="#007ACC", fg="white", font=("Helvetica", 12)).pack(pady=10)
    tk.Button(frame_izquierda, text="Limpiar Entradas", command=limpiar, bg="orange", font=("Helvetica", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Ver Tabla de Resultados", command=mostrar_tabla_resultados, bg="green", fg="white", font=("Helvetica", 11)).pack(pady=5)
    tk.Button(frame_izquierda, text="Volver al Menú", command=volver_menu, bg="gray", fg="white", font=("Helvetica", 11)).pack(pady=5)

    # FRAME DERECHO (SECCIÓN 1) - Gráfica
    frame_derecha = tk.Frame(root, bg="#2A2A2A", padx=10, pady=10)
    # Imagen de fondo para el frame derecho
    try:
        right_bg_img_path = os.path.join(script_dir, "assets", "right_bg.png")
        right_bg_img = Image.open(right_bg_img_path)
        right_bg_img = right_bg_img.resize((700, 600))
        right_bg_photo = ImageTk.PhotoImage(right_bg_img)
        right_bg_label = tk.Label(frame_derecha, image=right_bg_photo)
        right_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        right_bg_label.image = right_bg_photo
    except Exception:
        frame_derecha.configure(bg="#2A2A2A")
    frame_derecha.pack(side="right", fill="both", expand=True)

    root.mainloop()
# Iniciar menú principal
mostrar_menu()
